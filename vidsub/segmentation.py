"""Caption segmentation logic for converting transcript segments to subtitles."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vidsub.models import Segment, SubtitleConfig

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Caption:
    """A subtitle caption (may span multiple lines)."""

    start: float
    end: float
    lines: list[str]

    @property
    def text(self) -> str:
        """Return caption text with line breaks."""
        return "\\N".join(self.lines)


def split_text_into_lines(
    text: str,
    max_chars_per_line: int,
    max_lines: int,
) -> list[str] | None:
    """Split text into lines respecting constraints.

    Args:
        text: Input text to split.
        max_chars_per_line: Maximum characters per line.
        max_lines: Maximum number of lines.

    Returns:
        List of lines, or None if text cannot fit constraints.
    """
    # Clean text
    text = text.strip()
    if not text:
        return []

    # Try to fit without splitting
    if len(text) <= max_chars_per_line:
        return [text]

    # Try to split at punctuation
    lines = _split_at_punctuation(text, max_chars_per_line, max_lines)
    if lines:
        return lines

    # Try to split at word boundaries
    lines = _split_at_words(text, max_chars_per_line, max_lines)
    if lines:
        return lines

    # Text too long, cannot fit
    logger.debug(f"Text cannot fit: {len(text)} chars, max {max_chars_per_line * max_lines}")
    return None


def _split_at_punctuation(
    text: str,
    max_chars: int,
    max_lines: int,
) -> list[str] | None:
    """Split text at punctuation marks."""
    # Punctuation marks to split on (in priority order)
    punct_pattern = r'(?<=[.!?;:，。？！；：])\s+'

    parts = re.split(punct_pattern, text)
    lines = []
    current_line = ""

    for part in parts:
        part = part.strip()
        if not part:
            continue

        if len(part) > max_chars:
            # Part too long for single line, abort punctuation strategy
            return None

        if not current_line:
            current_line = part
        elif len(current_line) + 1 + len(part) <= max_chars:
            current_line += " " + part
        else:
            lines.append(current_line)
            if len(lines) >= max_lines:
                return None  # Too many lines needed
            current_line = part

    if current_line:
        lines.append(current_line)

    return lines if lines else None


def _split_at_words(
    text: str,
    max_chars: int,
    max_lines: int,
) -> list[str] | None:
    """Split text at word boundaries."""
    words = text.split()
    lines = []
    current_line = ""

    for word in words:
        if len(word) > max_chars:
            # Word too long, cannot split further
            return None

        if not current_line:
            current_line = word
        elif len(current_line) + 1 + len(word) <= max_chars:
            current_line += " " + word
        else:
            lines.append(current_line)
            if len(lines) >= max_lines:
                return None  # Too many lines needed
            current_line = word

    if current_line:
        lines.append(current_line)

    return lines if lines else None


def segment_to_captions(
    segment: Segment,
    config: SubtitleConfig,
    min_caption_duration: float = 1.0,
) -> list[Caption]:
    """Convert a transcript segment to one or more captions.

    Args:
        segment: Transcript segment.
        config: Subtitle configuration.
        min_caption_duration: Minimum caption duration in seconds.

    Returns:
        List of captions.
    """
    text = segment.text.strip()
    if not text:
        return []

    segment_duration = segment.end - segment.start
    max_duration = config.max_caption_seconds
    word_count = len(text.split())

    logger.debug(f"Processing segment: {word_count} words, "
                 f"{segment_duration:.2f}s, {len(text)} chars")

    # Try to fit entire segment in one caption
    lines = split_text_into_lines(
        text,
        config.max_chars_per_line,
        config.max_lines,
    )

    if lines and segment_duration <= max_duration:
        # Fits in one caption
        logger.debug(f"Segment fits in single caption ({len(lines)} lines)")
        return [Caption(
            start=segment.start,
            end=segment.end,
            lines=lines,
        )]

    # Need to split into multiple captions
    logger.debug(f"Segment requires splitting (duration={segment_duration:.2f}s, "
                 f"max={max_duration:.2f}s or text too long)")
    return _split_segment(segment, config, min_caption_duration)


def _split_segment(
    segment: Segment,
    config: SubtitleConfig,
    min_duration: float,
) -> list[Caption]:
    """Split a long segment into multiple captions."""
    logger.debug(f"Splitting segment: {len(segment.text)} chars, "
                 f"{segment.end - segment.start:.2f}s duration")

    captions = []
    words = segment.text.split()

    if not words:
        logger.debug("Empty segment, returning no captions")
        return []

    # Estimate timing per word
    segment_duration = segment.end - segment.start
    time_per_word = segment_duration / len(words)
    logger.debug(f"Time per word estimate: {time_per_word:.3f}s")

    current_words: list[str] = []
    current_start = segment.start
    processed_count = 0

    for word in words:
        # Test if adding this word would still fit
        test_words = current_words + [word] if current_words else [word]
        test_text = " ".join(test_words)
        lines = split_text_into_lines(
            test_text,
            config.max_chars_per_line,
            config.max_lines,
        )

        if lines is not None:
            # Word fits, add it to current buffer
            current_words.append(word)
            processed_count += 1
        else:
            # Word doesn't fit, finalize previous caption first
            if current_words:
                # Create caption from accumulated words
                text = " ".join(current_words)
                lines = split_text_into_lines(
                    text,
                    config.max_chars_per_line,
                    config.max_lines,
                )

                if lines:
                    # Calculate end time based on word count
                    word_count = len(current_words)
                    estimated_duration = word_count * time_per_word
                    end_time = min(
                        current_start + estimated_duration,
                        segment.end,
                    )

                    # Ensure minimum duration
                    if end_time - current_start < min_duration:
                        end_time = min(current_start + min_duration, segment.end)

                    captions.append(Caption(
                        start=current_start,
                        end=end_time,
                        lines=lines,
                    ))

                    # Start next caption
                    current_start = end_time

            # Now handle the new word that didn't fit
            # Check if the single word itself can fit
            single_word_lines = split_text_into_lines(
                word,
                config.max_chars_per_line,
                config.max_lines,
            )

            if single_word_lines:
                # Word fits on its own, start new caption with it
                current_words = [word]
                processed_count += 1
            else:
                # Single word too long, force it as a caption
                logger.debug(f"Word too long, forcing split: '{word[:20]}...'")
                lines = [word[:config.max_chars_per_line]]
                end_time = min(current_start + min_duration, segment.end)

                captions.append(Caption(
                    start=current_start,
                    end=end_time,
                    lines=lines,
                ))

                current_start = end_time
                current_words = []
                processed_count += 1

    # Handle remaining words
    if current_words:
        text = " ".join(current_words)
        lines = split_text_into_lines(
            text,
            config.max_chars_per_line,
            config.max_lines,
        )
        if not lines:
            lines = [text[:config.max_chars_per_line]]

        captions.append(Caption(
            start=current_start,
            end=segment.end,
            lines=lines,
        ))

    logger.debug(f"Segment split into {len(captions)} captions, "
                 f"processed {processed_count} words")

    # Adjust timing to ensure no gaps and proper bounds
    for i, caption in enumerate(captions):
        # Ensure caption doesn't exceed max duration
        if caption.end - caption.start > config.max_caption_seconds:
            caption_end = caption.start + config.max_caption_seconds
            captions[i] = Caption(
                start=caption.start,
                end=min(caption_end, segment.end),
                lines=caption.lines,
            )

    return captions


def merge_small_gaps(
    captions: list[Caption],
    max_gap_sec: float = 0.1,
) -> list[Caption]:
    """Merge captions with small gaps between them.

    Args:
        captions: List of captions.
        max_gap_sec: Maximum gap to merge (default 100ms).

    Returns:
        List with gaps merged.
    """
    if len(captions) < 2:
        return captions

    merged = [captions[0]]

    for caption in captions[1:]:
        prev = merged[-1]
        gap = caption.start - prev.end

        if 0 < gap <= max_gap_sec:
            # Extend previous caption to remove gap
            merged[-1] = Caption(
                start=prev.start,
                end=caption.start,
                lines=prev.lines,
            )

        merged.append(caption)

    return merged
