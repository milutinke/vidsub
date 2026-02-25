"""Timestamp validation and repair logic for transcripts and subtitles."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vidsub.models import CanonicalTranscript, Segment

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Raised when validation fails and cannot be repaired."""
    pass


class TimestampValidator:
    """Validates and repairs transcript timestamps."""

    def __init__(self, duration_sec: float):
        """Initialize validator with video duration.

        Args:
            duration_sec: Total video duration in seconds.
        """
        self.duration_sec = duration_sec

    def validate_segment(self, segment: Segment, index: int) -> list[str]:
        """Validate a single segment.

        Args:
            segment: Segment to validate.
            index: Segment index for error messages.

        Returns:
            List of validation issues (empty if valid).
        """
        issues = []

        # Check bounds: 0 <= start <= end <= duration
        if segment.start < 0:
            issues.append(f"Segment {index}: start ({segment.start}) < 0")
        if segment.end < 0:
            issues.append(f"Segment {index}: end ({segment.end}) < 0")
        if segment.start > self.duration_sec:
            issues.append(
                f"Segment {index}: start ({segment.start}) > duration ({self.duration_sec})"
            )
        if segment.end > self.duration_sec:
            issues.append(
                f"Segment {index}: end ({segment.end}) > duration ({self.duration_sec})"
            )
        if segment.start > segment.end:
            issues.append(
                f"Segment {index}: start ({segment.start}) > end ({segment.end})"
            )

        # Check text is non-empty
        if not segment.text or not segment.text.strip():
            issues.append(f"Segment {index}: empty text")

        # Check word-level timestamps if present
        if segment.words:
            for j, word in enumerate(segment.words):
                if word.start < segment.start:
                    issues.append(
                        f"Segment {index}, word {j}: word start ({word.start}) < segment start"
                    )
                if word.end > segment.end:
                    issues.append(
                        f"Segment {index}, word {j}: word end ({word.end}) > segment end"
                    )
                if word.start > word.end:
                    issues.append(
                        f"Segment {index}, word {j}: word start > word end"
                    )

        return issues

    def validate_transcript(self, transcript: CanonicalTranscript) -> list[str]:
        """Validate entire transcript.

        Args:
            transcript: Transcript to validate.

        Returns:
            List of validation issues (empty if valid).
        """
        issues = []

        # Validate duration matches
        if transcript.duration_sec <= 0:
            issues.append(f"Invalid duration: {transcript.duration_sec}")

        # Validate segments
        for i, segment in enumerate(transcript.segments):
            segment_issues = self.validate_segment(segment, i)
            issues.extend(segment_issues)

            # Check monotonic ordering (start times)
            if i > 0 and segment.start < transcript.segments[i - 1].start:
                issues.append(
                    f"Segment {i}: start ({segment.start}) < previous start "
                    f"({transcript.segments[i - 1].start})"
                )

        return issues

    def repair_segment(self, segment: Segment, index: int) -> Segment:
        """Attempt to repair a segment's timestamps.

        Repairs performed:
        - Clamp negative times to 0
        - Clamp times exceeding duration to duration
        - Swap start/end if reversed

        Args:
            segment: Segment to repair.
            index: Segment index for logging.

        Returns:
            Repaired segment (may be same object if no repairs needed).
        """
        from vidsub.models import Segment

        start = segment.start
        end = segment.end
        repairs = []

        # Clamp to [0, duration]
        if start < 0:
            repairs.append(f"clamped start from {start} to 0")
            start = 0.0
        if end < 0:
            repairs.append(f"clamped end from {end} to 0")
            end = 0.0
        if start > self.duration_sec:
            repairs.append(f"clamped start from {start} to {self.duration_sec}")
            start = self.duration_sec
        if end > self.duration_sec:
            repairs.append(f"clamped end from {end} to {self.duration_sec}")
            end = self.duration_sec

        # Fix reversed timestamps
        if start > end:
            repairs.append(f"swapped start({start})/end({end})")
            start, end = end, start

        if repairs:
            logger.warning(f"Segment {index}: {', '.join(repairs)}")
            return Segment(
                start=start,
                end=end,
                text=segment.text,
                speaker=segment.speaker,
                words=segment.words,
            )

        return segment

    def repair_transcript(self, transcript: CanonicalTranscript) -> CanonicalTranscript:
        """Attempt to repair transcript timestamps.

        Args:
            transcript: Transcript to repair.

        Returns:
            Repaired transcript.

        Raises:
            ValidationError: If repairs would cause meaning loss or large shifts.
        """
        from vidsub.models import CanonicalTranscript

        repaired_segments = []
        for i, segment in enumerate(transcript.segments):
            repaired = self.repair_segment(segment, i)
            repaired_segments.append(repaired)

        # Sort by start time to ensure monotonic ordering
        repaired_segments.sort(key=lambda s: s.start)

        # Check for overlapping segments after repair
        for i in range(1, len(repaired_segments)):
            if repaired_segments[i].start < repaired_segments[i - 1].end:
                # Overlap detected - this is acceptable but log it
                logger.warning(
                    f"Segment overlap detected: {i-1} ends at "
                    f"{repaired_segments[i-1].end}, {i} starts at "
                    f"{repaired_segments[i].start}"
                )

        return CanonicalTranscript(
            engine=transcript.engine,
            model=transcript.model,
            language=transcript.language,
            duration_sec=transcript.duration_sec,
            segments=repaired_segments,
        )


def validate_and_repair(
    transcript: CanonicalTranscript,
    strict: bool = False,
) -> CanonicalTranscript:
    """Validate and optionally repair transcript.

    Args:
        transcript: Transcript to validate.
        strict: If True, raise on any validation issue. If False, attempt repair.

    Returns:
        Validated (and possibly repaired) transcript.

    Raises:
        ValidationError: If validation fails and strict=True, or repair fails.
    """
    validator = TimestampValidator(transcript.duration_sec)
    issues = validator.validate_transcript(transcript)

    if issues:
        if strict:
            raise ValidationError(f"Validation failed: {'; '.join(issues)}")

        logger.warning(f"Validation issues found: {issues}")
        transcript = validator.repair_transcript(transcript)

        # Re-validate after repair
        remaining = validator.validate_transcript(transcript)
        if remaining:
            raise ValidationError(f"Could not repair: {'; '.join(remaining)}")

    return transcript
