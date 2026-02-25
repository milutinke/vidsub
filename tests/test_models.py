"""Tests for data models."""

import pytest
from pydantic import ValidationError

from vidsub.models import (
    AppConfig,
    CanonicalTranscript,
    Config,
    EngineConfig,
    Segment,
    Word,
)


class TestWord:
    """Tests for Word model."""

    def test_valid_word(self) -> None:
        word = Word(start=1.0, end=2.0, word="hello")
        assert word.start == 1.0
        assert word.end == 2.0
        assert word.word == "hello"

    def test_end_before_start_raises(self) -> None:
        with pytest.raises(ValidationError):
            Word(start=2.0, end=1.0, word="hello")

    def test_empty_word_raises(self) -> None:
        with pytest.raises(ValidationError):
            Word(start=1.0, end=2.0, word="")


class TestSegment:
    """Tests for Segment model."""

    def test_valid_segment(self) -> None:
        seg = Segment(start=0.0, end=5.0, text="Hello world")
        assert seg.start == 0.0
        assert seg.end == 5.0
        assert seg.text == "Hello world"

    def test_segment_with_words(self) -> None:
        words = [
            Word(start=0.0, end=1.0, word="Hello"),
            Word(start=1.0, end=2.0, word="world"),
        ]
        seg = Segment(start=0.0, end=2.0, text="Hello world", words=words)
        assert seg.words is not None
        assert len(seg.words) == 2


class TestCanonicalTranscript:
    """Tests for CanonicalTranscript model."""

    def test_valid_transcript(self) -> None:
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=60.0,
            segments=[
                Segment(start=0.0, end=5.0, text="Hello"),
                Segment(start=5.0, end=10.0, text="World"),
            ],
        )
        assert transcript.engine == "whisper"
        assert len(transcript.segments) == 2


class TestConfig:
    """Tests for Config model."""

    def test_default_config(self) -> None:
        config = Config()
        assert config.app.out_dir == "./out"
        assert config.engine.name == "whisper"
        assert config.whisper.model == "large"

    def test_config_with_overrides(self) -> None:
        config = Config(
            app=AppConfig(out_dir="./custom"),
            engine=EngineConfig(name="gemini"),
        )
        assert config.app.out_dir == "./custom"
        assert config.engine.name == "gemini"


class TestModelEdgeCases:
    """Edge case tests for data models."""

    def test_segment_with_whitespace_only_text(self) -> None:
        """Test segment with whitespace-only text is allowed but stripped."""
        # The model allows whitespace but validators may flag it
        seg = Segment(start=0.0, end=5.0, text="   ")
        assert seg.text == "   "

    def test_word_with_whitespace(self) -> None:
        """Test word with whitespace is allowed (phrase-level timing)."""
        # Words field allows phrases for phrase-level timing
        word = Word(start=0.0, end=1.0, word="hello world")
        assert word.word == "hello world"

    def test_transcript_with_zero_duration_fails(self) -> None:
        """Test transcript with zero duration fails validation."""
        with pytest.raises(ValidationError):
            CanonicalTranscript(
                engine="whisper",
                model="large",
                language="en",
                duration_sec=0.0,
                segments=[],
            )

    def test_transcript_with_very_long_duration(self) -> None:
        """Test transcript with very long duration (hours)."""
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=7200.0,  # 2 hours
            segments=[Segment(start=0.0, end=7199.0, text="Long video")],
        )
        assert transcript.duration_sec == 7200.0

    def test_segment_with_microsecond_precision(self) -> None:
        """Test segment with microsecond-level timestamps."""
        seg = Segment(start=1.123456, end=2.987654, text="Precise")
        assert abs(seg.start - 1.123456) < 0.000001
        assert abs(seg.end - 2.987654) < 0.000001

    def test_segment_with_unicode_text(self) -> None:
        """Test segment with unicode characters."""
        seg = Segment(
            start=0.0,
            end=5.0,
            text="Hello 世界 👋 ñoño café naïve",
        )
        assert "世界" in seg.text
        assert "👋" in seg.text
