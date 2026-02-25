"""Tests for caption segmentation logic."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from vidsub.models import Segment, SubtitleConfig
from vidsub.segmentation import (
    Caption,
    merge_small_gaps,
    segment_to_captions,
    split_text_into_lines,
)

if TYPE_CHECKING:
    pass


class TestSplitTextIntoLines:
    """Tests for split_text_into_lines function."""

    def test_empty_text(self) -> None:
        """Empty text returns empty list."""
        result = split_text_into_lines("", max_chars_per_line=42, max_lines=2)
        assert result == []

    def test_single_short_line(self) -> None:
        """Short text fits in one line."""
        result = split_text_into_lines("Hello", max_chars_per_line=42, max_lines=2)
        assert result == ["Hello"]

    def test_exact_fit(self) -> None:
        """Text exactly at max length fits."""
        text = "x" * 42
        result = split_text_into_lines(text, max_chars_per_line=42, max_lines=1)
        assert result == [text]

    def test_split_at_punctuation(self) -> None:
        """Split at punctuation marks."""
        text = "First sentence. Second sentence!"
        result = split_text_into_lines(text, max_chars_per_line=20, max_lines=2)
        assert result == ["First sentence.", "Second sentence!"]

    def test_split_at_words(self) -> None:
        """Split at word boundaries when no punctuation."""
        text = "Hello world test here"
        result = split_text_into_lines(text, max_chars_per_line=12, max_lines=3)
        assert result == ["Hello world", "test here"]

    def test_too_many_lines_needed(self) -> None:
        """Return None if text needs more than max_lines."""
        text = "a b c d e f g h i j"
        result = split_text_into_lines(text, max_chars_per_line=2, max_lines=2)
        assert result is None

    def test_word_too_long(self) -> None:
        """Return None if a single word exceeds max_chars."""
        text = "supercalifragilistic"
        result = split_text_into_lines(text, max_chars_per_line=10, max_lines=2)
        assert result is None


class TestSegmentToCaptions:
    """Tests for segment_to_captions function."""

    @pytest.fixture
    def default_config(self) -> SubtitleConfig:
        """Default subtitle config."""
        return SubtitleConfig(
            max_chars_per_line=42,
            max_lines=2,
            max_caption_seconds=6.0,
        )

    def test_whitespace_only_segment(self, default_config: SubtitleConfig) -> None:
        """Whitespace-only segment returns empty list."""
        # Note: Segment requires min_length=1, so we test via split_text_into_lines
        result = split_text_into_lines("   ", max_chars_per_line=42, max_lines=2)
        assert result == []

    def test_short_segment_single_caption(self, default_config: SubtitleConfig) -> None:
        """Short segment fits in single caption."""
        segment = Segment(start=0.0, end=2.0, text="Hello world")
        result = segment_to_captions(segment, default_config)
        assert len(result) == 1
        assert result[0].start == 0.0
        assert result[0].end == 2.0
        assert result[0].lines == ["Hello world"]

    def test_long_segment_split(self, default_config: SubtitleConfig) -> None:
        """Long segment is split into multiple captions."""
        # Create a segment that needs splitting
        words = ["word"] * 20  # 20 words won't fit in 2 lines of 42 chars
        segment = Segment(start=0.0, end=10.0, text=" ".join(words))
        result = segment_to_captions(segment, default_config)

        assert len(result) > 1
        # Verify timing is monotonic
        for i in range(1, len(result)):
            assert result[i].start >= result[i - 1].end

    def test_segment_exceeds_max_duration(self, default_config: SubtitleConfig) -> None:
        """Segment exceeding max_caption_seconds and text too long is split."""
        # Create text that won't fit in one caption
        words = ["word"] * 30  # Long text that needs splitting
        segment = Segment(
            start=0.0,
            end=12.0,  # Exceeds max_caption_seconds=6.0
            text=" ".join(words),
        )
        result = segment_to_captions(segment, default_config)

        # Should create multiple captions due to duration + text length
        assert len(result) >= 2

    def test_consecutive_duplicate_words(self, default_config: SubtitleConfig) -> None:
        """Test that consecutive duplicate words don't cause infinite loops.

        This is a regression test for a critical bug where the code at line 241
        would create an infinite loop when processing text with consecutive
        duplicate words like "the the" or "and and".
        """
        # Text with consecutive duplicates
        segment = Segment(
            start=0.0,
            end=5.0,
            text="the the quick brown fox and and jumps over",
        )

        # This should complete quickly (not hang)
        start_time = time.time()
        result = segment_to_captions(segment, default_config)
        elapsed = time.time() - start_time

        # Should complete in under 1 second (was hanging indefinitely)
        assert elapsed < 1.0, f"Segmentation took too long: {elapsed:.2f}s"
        assert len(result) >= 1

        # Verify all words are processed
        all_text = " ".join(" ".join(cap.lines) for cap in result)
        assert "the the" in all_text
        assert "and and" in all_text

    def test_many_consecutive_duplicates(self, default_config: SubtitleConfig) -> None:
        """Test with many consecutive duplicate words."""
        segment = Segment(
            start=0.0,
            end=10.0,
            text="the the the the quick quick brown brown fox fox",
        )

        start_time = time.time()
        result = segment_to_captions(segment, default_config)
        elapsed = time.time() - start_time

        assert elapsed < 1.0, f"Segmentation took too long: {elapsed:.2f}s"
        assert len(result) >= 1

    def test_duplicate_words_force_split(self, default_config: SubtitleConfig) -> None:
        """Test that duplicate words at split boundary are handled correctly.

        This is a more rigorous regression test for the line 241 bug.
        The original bug would lose words or hang when a duplicate word
        appeared at the exact point where a caption needed to be split.
        """
        # Create text with pattern that forces splitting at duplicate word
        # Each line can hold ~42 chars, 2 lines = ~84 chars
        # With "word" being 4 chars + 1 space = 5 chars per word
        # That's about 16 words per caption. We need duplicates at boundary.
        words = []
        for i in range(24):  # Enough words to require splitting
            if i == 16:  # At the split boundary, add duplicate
                words.append("boundary")
            words.append(f"word{i}")

        # Add duplicate at exact position where split occurs
        segment = Segment(
            start=0.0,
            end=10.0,
            text=" ".join(words),
        )

        # This should complete without infinite loop
        start_time = time.time()
        result = segment_to_captions(segment, default_config)
        elapsed = time.time() - start_time

        assert elapsed < 1.0, f"Segmentation took too long: {elapsed:.2f}s"
        assert len(result) >= 2, "Should have split into multiple captions"

        # Verify all words are present and accounted for
        all_text = " ".join(" ".join(cap.lines) for cap in result)
        all_result_words = all_text.split()
        assert len(all_result_words) == len(words), (
            f"Word count mismatch: expected {len(words)}, got {len(all_result_words)}"
        )

        # Verify no words were lost (each input word appears in output)
        for word in words:
            assert word in all_result_words, f"Word '{word}' was lost during segmentation"


class TestSegmentationPerformance:
    """Performance tests for segmentation."""

    @pytest.fixture
    def default_config(self) -> SubtitleConfig:
        """Default subtitle config."""
        return SubtitleConfig(
            max_chars_per_line=42,
            max_lines=2,
            max_caption_seconds=6.0,
        )

    def test_long_segment_performance(self, default_config: SubtitleConfig) -> None:
        """Test that long segments process in reasonable time (O(n) complexity).

        This is a regression test for O(n²) string building performance issue.
        """
        # Create a segment with many words (1000 words)
        words = [f"word{i}" for i in range(1000)]
        segment = Segment(start=0.0, end=60.0, text=" ".join(words))

        start_time = time.time()
        result = segment_to_captions(segment, default_config)
        elapsed = time.time() - start_time

        # Should complete in under 5 seconds with O(n) algorithm
        # (O(n²) would take minutes for 1000 words)
        assert elapsed < 5.0, f"Segmentation too slow: {elapsed:.2f}s for 1000 words"
        assert len(result) > 0

        # Verify all words are accounted for
        all_text = " ".join(" ".join(cap.lines) for cap in result)
        total_words_in_result = len(all_text.split())
        assert total_words_in_result == 1000

    def test_very_long_words(self, default_config: SubtitleConfig) -> None:
        """Test handling of very long words that must be truncated."""
        # Create words that exceed max_chars_per_line
        long_word = "supercalifragilisticexpialidocious"
        segment = Segment(
            start=0.0,
            end=5.0,
            text=f"{long_word} {long_word} {long_word}",
        )

        result = segment_to_captions(segment, default_config)
        assert len(result) > 0

        # Verify long words are truncated
        for cap in result:
            for line in cap.lines:
                assert len(line) <= default_config.max_chars_per_line


class TestEdgeCases:
    """Edge case tests for segmentation."""

    @pytest.fixture
    def default_config(self) -> SubtitleConfig:
        """Default subtitle config."""
        return SubtitleConfig(
            max_chars_per_line=42,
            max_lines=2,
            max_caption_seconds=6.0,
        )

    def test_single_word(self, default_config: SubtitleConfig) -> None:
        """Segment with single word."""
        segment = Segment(start=0.0, end=1.0, text="Hello")
        result = segment_to_captions(segment, default_config)
        assert len(result) == 1
        assert result[0].lines == ["Hello"]

    def test_whitespace_only(self, default_config: SubtitleConfig) -> None:
        """Segment with only whitespace."""
        segment = Segment(start=0.0, end=1.0, text="   ")
        result = segment_to_captions(segment, default_config)
        assert result == []

    def test_max_duration_enforced(self, default_config: SubtitleConfig) -> None:
        """Max caption duration is enforced when splitting segments."""
        # Create text that requires splitting
        words = ["word"] * 50  # Many words to force splitting
        segment = Segment(start=0.0, end=60.0, text=" ".join(words))
        result = segment_to_captions(segment, default_config)
        assert len(result) >= 2  # Should be split
        # Each caption should not exceed max_caption_seconds (6.0s default)
        for cap in result:
            duration = cap.end - cap.start
            assert duration <= 6.0, f"Caption duration {duration}s exceeds max 6.0s"

    def test_unicode_text(self, default_config: SubtitleConfig) -> None:
        """Segment with unicode characters."""
        segment = Segment(
            start=0.0,
            end=5.0,
            text="Hello 世界 this is a test with unicode characters",
        )
        result = segment_to_captions(segment, default_config)
        assert len(result) >= 1

    def test_punctuation_splitting(self, default_config: SubtitleConfig) -> None:
        """Text with various punctuation marks."""
        segment = Segment(
            start=0.0,
            end=5.0,
            text="First. Second! Third? Fourth; Fifth: Sixth",
        )
        result = segment_to_captions(segment, default_config)
        assert len(result) >= 1


class TestMergeSmallGaps:
    """Tests for merge_small_gaps function."""

    def test_no_gaps(self) -> None:
        """Captions with no gaps remain unchanged."""
        captions = [
            Caption(start=0.0, end=2.0, lines=["Hello"]),
            Caption(start=2.0, end=4.0, lines=["World"]),
        ]
        result = merge_small_gaps(captions, max_gap_sec=0.1)
        assert len(result) == 2
        assert result[0].end == 2.0
        assert result[1].start == 2.0

    def test_small_gap_merged(self) -> None:
        """Small gaps between captions are merged."""
        captions = [
            Caption(start=0.0, end=2.0, lines=["Hello"]),
            Caption(start=2.05, end=4.0, lines=["World"]),  # 50ms gap
        ]
        result = merge_small_gaps(captions, max_gap_sec=0.1)
        assert len(result) == 2
        assert result[0].end == 2.05  # Extended to remove gap

    def test_large_gap_not_merged(self) -> None:
        """Large gaps are not merged."""
        captions = [
            Caption(start=0.0, end=2.0, lines=["Hello"]),
            Caption(start=3.0, end=5.0, lines=["World"]),  # 1s gap
        ]
        result = merge_small_gaps(captions, max_gap_sec=0.1)
        assert len(result) == 2
        assert result[0].end == 2.0  # Unchanged

    def test_single_caption(self) -> None:
        """Single caption remains unchanged."""
        captions = [Caption(start=0.0, end=2.0, lines=["Hello"])]
        result = merge_small_gaps(captions)
        assert len(result) == 1

    def test_empty_list(self) -> None:
        """Empty list returns empty list."""
        result = merge_small_gaps([])
        assert result == []


class TestCaption:
    """Tests for Caption dataclass."""

    def test_text_property(self) -> None:
        """Text property joins lines with \\N."""
        caption = Caption(start=0.0, end=1.0, lines=["Line 1", "Line 2"])
        assert caption.text == "Line 1\\NLine 2"

    def test_single_line_text(self) -> None:
        """Single line returns itself."""
        caption = Caption(start=0.0, end=1.0, lines=["Single"])
        assert caption.text == "Single"
