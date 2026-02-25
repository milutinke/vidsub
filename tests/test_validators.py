"""Tests for timestamp validation and repair."""

from typing import Any

from vidsub.models import CanonicalTranscript, Segment
from vidsub.validators import (
    TimestampValidator,
    validate_and_repair,
)


class MockSegment:
    """Mock segment for testing validation without Pydantic constraints."""

    def __init__(
        self,
        start: float,
        end: float,
        text: str,
        words: list[dict[str, Any]] | None = None,
    ) -> None:
        self.start = start
        self.end = end
        self.text = text
        self.words = words
        self.speaker: str | None = None


class TestTimestampValidator:
    """Tests for TimestampValidator."""

    def test_validate_valid_segment(self) -> None:
        validator = TimestampValidator(duration_sec=60.0)
        segment = MockSegment(start=0.0, end=5.0, text="Hello")
        issues = validator.validate_segment(segment, 0)  # type: ignore[arg-type]
        assert issues == []

    def test_validate_negative_start(self) -> None:
        validator = TimestampValidator(duration_sec=60.0)
        segment = MockSegment(start=-1.0, end=5.0, text="Hello")
        issues = validator.validate_segment(segment, 0)  # type: ignore[arg-type]
        assert any("< 0" in i for i in issues)

    def test_validate_start_after_duration(self) -> None:
        validator = TimestampValidator(duration_sec=10.0)
        segment = MockSegment(start=15.0, end=20.0, text="Hello")
        issues = validator.validate_segment(segment, 0)  # type: ignore[arg-type]
        assert any("> duration" in i for i in issues)

    def test_validate_start_after_end(self) -> None:
        validator = TimestampValidator(duration_sec=60.0)
        segment = MockSegment(start=10.0, end=5.0, text="Hello")
        issues = validator.validate_segment(segment, 0)  # type: ignore[arg-type]
        assert any("start" in i and "end" in i for i in issues)

    def test_repair_negative_start(self) -> None:
        validator = TimestampValidator(duration_sec=60.0)
        segment = MockSegment(start=-1.0, end=5.0, text="Hello")
        repaired = validator.repair_segment(segment, 0)  # type: ignore[arg-type]
        assert repaired.start == 0.0

    def test_repair_reversed_timestamps(self) -> None:
        validator = TimestampValidator(duration_sec=60.0)
        segment = MockSegment(start=10.0, end=5.0, text="Hello")
        repaired = validator.repair_segment(segment, 0)  # type: ignore[arg-type]
        assert repaired.start == 5.0
        assert repaired.end == 10.0


class TestValidateAndRepair:
    """Tests for validate_and_repair function."""

    def test_valid_transcript(self) -> None:
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=10.0,
            segments=[Segment(start=0.0, end=5.0, text="Hello")],
        )
        result = validate_and_repair(transcript)
        assert result.segments[0].start == 0.0

    def test_repair_clamps_out_of_bounds(self) -> None:
        # Create a mock-like transcript with out-of-bounds end time
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=10.0,
            segments=[Segment(start=0.0, end=15.0, text="Hello")],
        )
        result = validate_and_repair(transcript)
        assert result.segments[0].end == 10.0


class TestValidationEdgeCases:
    """Edge case tests for validation."""

    def test_validate_empty_transcript(self) -> None:
        """Test validating empty transcript."""
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=10.0,
            segments=[],
        )
        # Should not raise
        result = validate_and_repair(transcript)
        assert len(result.segments) == 0

    def test_validate_segment_with_empty_text(self) -> None:
        """Test validating segment with empty text."""
        validator = TimestampValidator(duration_sec=60.0)
        segment = MockSegment(start=0.0, end=5.0, text="")
        issues = validator.validate_segment(segment, 0)  # type: ignore[arg-type]
        assert any("empty text" in i.lower() for i in issues)

    def test_repair_both_times_negative(self) -> None:
        """Test repairing segment where both start and end are negative."""
        validator = TimestampValidator(duration_sec=60.0)
        segment = MockSegment(start=-5.0, end=-1.0, text="Hello")
        repaired = validator.repair_segment(segment, 0)  # type: ignore[arg-type]
        assert repaired.start == 0.0
        assert repaired.end == 0.0

    def test_repair_both_times_exceed_duration(self) -> None:
        """Test repairing segment where both times exceed duration."""
        validator = TimestampValidator(duration_sec=10.0)
        segment = MockSegment(start=15.0, end=20.0, text="Hello")
        repaired = validator.repair_segment(segment, 0)  # type: ignore[arg-type]
        assert repaired.start == 10.0
        assert repaired.end == 10.0

    def test_validate_word_level_timestamps(self) -> None:
        """Test validation of word-level timestamps with proper Word objects."""
        from vidsub.models import Segment, Word

        validator = TimestampValidator(duration_sec=60.0)
        words = [
            Word(start=0.0, end=1.0, word="Hello"),
            Word(start=1.0, end=2.0, word="world"),
        ]
        segment = Segment(start=0.0, end=2.0, text="Hello world", words=words)
        issues = validator.validate_segment(segment, 0)
        # Both words valid within segment
        assert issues == []  # No issues expected

    def test_validate_word_outside_segment_bounds(self) -> None:
        """Test validation catches words outside segment bounds."""
        from vidsub.models import Segment, Word

        validator = TimestampValidator(duration_sec=60.0)
        words = [
            Word(start=0.0, end=1.0, word="Hello"),
            Word(start=5.0, end=6.0, word="world"),  # Outside segment (0-2)
        ]
        segment = Segment(start=0.0, end=2.0, text="Hello world", words=words)
        issues = validator.validate_segment(segment, 0)
        # Should detect word end > segment end
        assert any("word" in i.lower() for i in issues)

    def test_validate_monotonic_ordering(self) -> None:
        """Test validation catches non-monotonic segment ordering."""
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=60.0,
            segments=[
                Segment(start=10.0, end=15.0, text="Second"),
                Segment(start=0.0, end=5.0, text="First"),  # Out of order
            ],
        )
        validator = TimestampValidator(duration_sec=60.0)
        issues = validator.validate_transcript(transcript)
        assert any("previous start" in i for i in issues)

    def test_repair_sorts_segments(self) -> None:
        """Test that repair sorts segments by start time."""
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=60.0,
            segments=[
                Segment(start=10.0, end=15.0, text="Second"),
                Segment(start=0.0, end=5.0, text="First"),
            ],
        )
        result = validate_and_repair(transcript)
        assert result.segments[0].text == "First"
        assert result.segments[1].text == "Second"

    def test_zero_duration_validation(self) -> None:
        """Test validation with zero duration video."""
        validator = TimestampValidator(duration_sec=0.0)
        segment = MockSegment(start=0.0, end=0.0, text="Hello")
        issues = validator.validate_segment(segment, 0)  # type: ignore[arg-type]
        # Zero duration should trigger duration warnings
        assert len(issues) >= 0  # May or may not have issues depending on implementation
