"""Pipeline orchestration for transcribe + subtitle + burn workflow."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vidsub.burn import burn_subtitles
from vidsub.engines.base import EngineFactory
from vidsub.models import CanonicalTranscript
from vidsub.subtitles import write_ass, write_srt
from vidsub.validators import validate_and_repair

if TYPE_CHECKING:
    from vidsub.models import Config

logger = logging.getLogger(__name__)


class PipelineError(Exception):
    """Exception for pipeline failures."""
    pass


class PipelineResult:
    """Result of pipeline execution."""

    def __init__(
        self,
        transcript: CanonicalTranscript,
        transcript_path: Path | None = None,
        srt_path: Path | None = None,
        ass_path: Path | None = None,
        burned_video_path: Path | None = None,
    ):
        self.transcript = transcript
        self.transcript_path = transcript_path
        self.srt_path = srt_path
        self.ass_path = ass_path
        self.burned_video_path = burned_video_path

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for reporting."""
        return {
            "transcript_path": str(self.transcript_path) if self.transcript_path else None,
            "srt_path": str(self.srt_path) if self.srt_path else None,
            "ass_path": str(self.ass_path) if self.ass_path else None,
            "burned_video_path": str(self.burned_video_path) if self.burned_video_path else None,
        }


def run_pipeline(
    video_path: Path,
    config: Config,
    skip_transcribe: bool = False,
    existing_transcript: Path | None = None,
    preset_name: str | None = None,
) -> PipelineResult:
    """Run complete transcription and subtitle pipeline.

    Args:
        video_path: Input video file.
        config: Application configuration.
        skip_transcribe: If True, use existing transcript instead of transcribing.
        existing_transcript: Path to existing transcript JSON (used if skip_transcribe=True).
        preset_name: Optional FFmpeg preset name for burning.

    Returns:
        PipelineResult with all generated files.
    """
    # Create output directory
    out_dir = Path(config.app.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Transcription (or load existing)
    if skip_transcribe and existing_transcript:
        logger.info(f"Loading existing transcript: {existing_transcript}")
        transcript = _load_transcript(existing_transcript)
    else:
        logger.info(f"Transcribing with {config.engine.name} engine")
        transcript = _transcribe(video_path, config)

    # Step 2: Validate and repair
    logger.info("Validating transcript")
    transcript = validate_and_repair(transcript)

    # Step 3: Write transcript JSON
    transcript_path = out_dir / f"{video_path.stem}_transcript.json"
    _write_transcript(transcript, transcript_path, config)

    # Step 4: Generate subtitles
    srt_path = None
    ass_path = None

    if "srt" in config.subtitles.formats or "both" in config.subtitles.formats:
        srt_path = out_dir / f"{video_path.stem}.srt"
        logger.info(f"Writing SRT: {srt_path}")
        write_srt(transcript, config.subtitles, srt_path)

    if "ass" in config.subtitles.formats or "both" in config.subtitles.formats:
        ass_path = out_dir / f"{video_path.stem}.ass"
        logger.info(f"Writing ASS: {ass_path}")
        write_ass(transcript, config.subtitles, config.style_ass, ass_path)

    # Step 5: Burn subtitles if requested
    burned_path = None
    if config.app.burn:
        subtitle_to_burn = ass_path if ass_path else srt_path
        if subtitle_to_burn:
            burned_path = out_dir / f"{video_path.stem}_burned.mp4"
            logger.info(f"Burning subtitles: {burned_path}")
            burn_subtitles(video_path, subtitle_to_burn, burned_path, config, preset_name=preset_name)

    # Step 6: Write run report
    report_path = out_dir / f"{video_path.stem}_report.json"
    _write_report(
        report_path,
        video_path,
        config,
        transcript,
        PipelineResult(
            transcript=transcript,
            transcript_path=transcript_path,
            srt_path=srt_path,
            ass_path=ass_path,
            burned_video_path=burned_path,
        ),
    )

    return PipelineResult(
        transcript=transcript,
        transcript_path=transcript_path,
        srt_path=srt_path,
        ass_path=ass_path,
        burned_video_path=burned_path,
    )


def _transcribe(video_path: Path, config: Config) -> CanonicalTranscript:
    """Run transcription with configured engine.

    Args:
        video_path: Input video.
        config: Configuration.

    Returns:
        Canonical transcript.
    """
    engine = EngineFactory.create(config)
    return engine.transcribe(video_path)


def _load_transcript(transcript_path: Path) -> CanonicalTranscript:
    """Load transcript from JSON file.

    Args:
        transcript_path: Path to transcript JSON.

    Returns:
        CanonicalTranscript.
    """
    data = json.loads(transcript_path.read_text())
    return CanonicalTranscript.model_validate(data)


def _write_transcript(
    transcript: CanonicalTranscript,
    output_path: Path,
    config: Config,
) -> None:
    """Write transcript to JSON file.

    Args:
        transcript: Transcript to write.
        output_path: Output file path.
        config: Configuration.
    """
    if output_path.exists() and not config.app.overwrite:
        raise PipelineError(f"Output exists (use --overwrite): {output_path}")

    output_path.write_text(
        transcript.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info(f"Wrote transcript: {output_path}")


def _write_report(
    report_path: Path,
    video_path: Path,
    config: Config,
    transcript: CanonicalTranscript,
    result: PipelineResult,
) -> None:
    """Write run report JSON.

    Args:
        report_path: Path for report file.
        video_path: Input video path.
        config: Configuration used.
        transcript: Generated transcript.
        result: Pipeline result.
    """
    report = {
        "input": {
            "video_path": str(video_path),
            "video_name": video_path.name,
        },
        "config": {
            "engine": config.engine.name,
            "model": _get_model_name(config),
            "language": config.engine.language,
        },
        "output": result.to_dict(),
        "transcript_stats": {
            "duration_sec": transcript.duration_sec,
            "segment_count": len(transcript.segments),
            "word_count": sum(
                len(s.words) if s.words else len(s.text.split())
                for s in transcript.segments
            ),
        },
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(f"Wrote report: {report_path}")


def _get_model_name(config: Config) -> str:
    """Get model name from config."""
    if config.engine.name == "whisper":
        return config.whisper.model
    elif config.engine.name == "gemini":
        return config.gemini.model
    return "unknown"
