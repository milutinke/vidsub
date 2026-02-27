"""CLI entry points for vidsub."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer

from vidsub.config import load_config, write_config_template
from vidsub.pipeline import PipelineError, run_pipeline

app = typer.Typer(
    name="vidsub",
    help="CLI tool for video subtitling with dual-engine transcription",
    no_args_is_help=True,
)

# Shared option types
EngineOption = Annotated[
    str | None,
    typer.Option(
        "--engine",
        help="Transcription engine: whisper or gemini",
    ),
]

ConfigOption = Annotated[
    Path | None,
    typer.Option(
        "--config",
        "-c",
        help="Path to config file",
        exists=True,
        file_okay=True,
        dir_okay=False,
    ),
]

OutDirOption = Annotated[
    Path | None,
    typer.Option(
        "--out-dir",
        "-o",
        help="Output directory",
        file_okay=False,
        dir_okay=True,
    ),
]

TempDirOption = Annotated[
    Path | None,
    typer.Option(
        "--temp-dir",
        "-t",
        help="Temporary directory",
        file_okay=False,
        dir_okay=True,
    ),
]

LanguageOption = Annotated[
    str | None,
    typer.Option(
        "--language",
        "-l",
        help="Language code (e.g., en, es)",
    ),
]

WhisperModelOption = Annotated[
    str | None,
    typer.Option(
        "--whisper-model",
        help="Whisper model alias, local path, or Hugging Face repo id",
    ),
]

FormatOption = Annotated[
    str,
    typer.Option(
        "--format",
        "-f",
        help="Subtitle format: srt, ass, or both",
    ),
]

BurnOption = Annotated[
    bool,
    typer.Option(
        "--burn/--no-burn",
        help="Burn subtitles into output video",
    ),
]

OverwriteOption = Annotated[
    bool,
    typer.Option(
        "--overwrite",
        help="Overwrite existing files",
    ),
]

LogLevelOption = Annotated[
    str,
    typer.Option(
        "--log-level",
        help="Logging level: debug, info, warning, error",
    ),
]

KeepTempOption = Annotated[
    bool,
    typer.Option(
        "--keep-temp",
        help="Keep temporary files for debugging",
    ),
]

SubtitleBgOption = Annotated[
    str | None,
    typer.Option(
        "--subtitle-bg",
        help="Subtitle background: none or solid",
    ),
]

VerboseSegOption = Annotated[
    bool,
    typer.Option(
        "--verbose-segmentation",
        help="Enable verbose logging for segmentation/post-processing",
    ),
]

ProgressOption = Annotated[
    bool,
    typer.Option(
        "--progress/--no-progress",
        help="Show/hide progress bars during processing",
    ),
]

PresetOption = Annotated[
    str | None,
    typer.Option(
        "--preset",
        "-p",
        help="FFmpeg encoding preset from config",
    ),
]


def _build_overrides(
    engine: str | None = None,
    out_dir: Path | None = None,
    temp_dir: Path | None = None,
    language: str | None = None,
    whisper_model: str | None = None,
    subtitle_bg: str | None = None,
    overwrite: bool | None = None,
    keep_temp: bool | None = None,
    burn: bool | None = None,
    verbose_segmentation: bool | None = None,
    progress: bool | None = None,
    preset: str | None = None,
) -> dict[str, Any] | None:
    """Build override dict from CLI options."""
    overrides: dict[str, Any] = {}

    if engine:
        overrides["engine"] = {"name": engine}
    if out_dir:
        overrides["app"] = overrides.get("app", {})
        overrides["app"]["out_dir"] = str(out_dir)
    if temp_dir:
        overrides["app"] = overrides.get("app", {})
        overrides["app"]["temp_dir"] = str(temp_dir)
    if language:
        overrides["engine"] = overrides.get("engine", {})
        overrides["engine"]["language"] = language
    if whisper_model:
        overrides["whisper"] = overrides.get("whisper", {})
        overrides["whisper"]["model"] = whisper_model
    if subtitle_bg:
        overrides["subtitles"] = overrides.get("subtitles", {})
        overrides["subtitles"]["background_style"] = subtitle_bg
    if overwrite is not None:
        overrides["app"] = overrides.get("app", {})
        overrides["app"]["overwrite"] = overwrite
    if keep_temp is not None:
        overrides["app"] = overrides.get("app", {})
        overrides["app"]["keep_temp"] = keep_temp
    if burn is not None:
        overrides["app"] = overrides.get("app", {})
        overrides["app"]["burn"] = burn
    if verbose_segmentation is not None:
        overrides["app"] = overrides.get("app", {})
        overrides["app"]["verbose_postprocessing"] = verbose_segmentation
    if progress is not None:
        overrides["app"] = overrides.get("app", {})
        overrides["app"]["show_progress"] = progress
    if preset is not None:
        overrides["ffmpeg"] = overrides.get("ffmpeg", {})
        overrides["ffmpeg"]["default_preset"] = preset

    return overrides if overrides else None


def _validate_engine_options(engine_name: str, whisper_model: str | None) -> None:
    """Validate options specific to the selected transcription engine."""
    if whisper_model and engine_name != "whisper":
        raise typer.BadParameter(
            f"whisper model overrides require the whisper engine; current engine: {engine_name}.",
            param_hint="--whisper-model",
        )


@app.command()
def run(
    video: Annotated[Path, typer.Argument(help="Input video file", exists=True)],
    engine: EngineOption = None,
    config: ConfigOption = None,
    out_dir: OutDirOption = None,
    temp_dir: TempDirOption = None,
    language: LanguageOption = None,
    whisper_model: WhisperModelOption = None,
    format: FormatOption = "both",
    burn: BurnOption = False,
    overwrite: OverwriteOption = False,
    log_level: LogLevelOption = "info",
    keep_temp: KeepTempOption = False,
    subtitle_bg: SubtitleBgOption = None,
    verbose_segmentation: VerboseSegOption = False,
    progress: ProgressOption = True,
    preset: PresetOption = None,
) -> None:
    """Run complete pipeline: transcribe, generate subtitles, optionally burn."""
    # Setup logging
    _setup_logging(log_level)

    # Adjust log level for verbose segmentation
    if verbose_segmentation and log_level == "info":
        _setup_logging("debug")

    overrides = _build_overrides(
        engine=engine,
        out_dir=out_dir,
        temp_dir=temp_dir,
        language=language,
        whisper_model=whisper_model,
        subtitle_bg=subtitle_bg,
        overwrite=overwrite,
        keep_temp=keep_temp,
        burn=burn,
        verbose_segmentation=verbose_segmentation,
        progress=progress,
        preset=preset,
    )
    cfg = load_config(config, overrides)
    _validate_engine_options(cfg.engine.name, whisper_model)

    typer.echo(f"Processing: {video}")
    typer.echo(f"Engine: {cfg.engine.name}")
    typer.echo(f"Output directory: {cfg.app.out_dir}")

    try:
        result = run_pipeline(video, cfg, preset_name=preset)
        typer.echo("\n✓ Processing complete!")
        typer.echo(f"  Transcript: {result.transcript_path}")
        if result.srt_path:
            typer.echo(f"  SRT: {result.srt_path}")
        if result.ass_path:
            typer.echo(f"  ASS: {result.ass_path}")
        if result.burned_video_path:
            typer.echo(f"  Burned video: {result.burned_video_path}")
    except PipelineError as e:
        typer.echo(f"\n✗ Error: {e}", err=True)
        raise typer.Exit(1) from None


@app.command()
def transcribe(
    video: Annotated[Path, typer.Argument(help="Input video file", exists=True)],
    engine: EngineOption = None,
    config: ConfigOption = None,
    out_dir: OutDirOption = None,
    temp_dir: TempDirOption = None,
    language: LanguageOption = None,
    whisper_model: WhisperModelOption = None,
    format: FormatOption = "both",
    overwrite: OverwriteOption = False,
    log_level: LogLevelOption = "info",
    keep_temp: KeepTempOption = False,
    verbose_segmentation: VerboseSegOption = False,
    progress: ProgressOption = True,
) -> None:
    """Transcribe video and generate subtitle files."""
    _setup_logging(log_level)

    # Adjust log level for verbose segmentation
    if verbose_segmentation and log_level == "info":
        _setup_logging("debug")

    overrides = _build_overrides(
        engine=engine,
        out_dir=out_dir,
        temp_dir=temp_dir,
        language=language,
        whisper_model=whisper_model,
        overwrite=overwrite,
        keep_temp=keep_temp,
        burn=False,
        verbose_segmentation=verbose_segmentation,
        progress=progress,
    )
    cfg = load_config(config, overrides)
    _validate_engine_options(cfg.engine.name, whisper_model)

    typer.echo(f"Transcribing: {video}")
    typer.echo(f"Engine: {cfg.engine.name}")

    try:
        result = run_pipeline(video, cfg)
        typer.echo("\n✓ Transcription complete!")
        typer.echo(f"  Transcript: {result.transcript_path}")
        if result.srt_path:
            typer.echo(f"  SRT: {result.srt_path}")
        if result.ass_path:
            typer.echo(f"  ASS: {result.ass_path}")
    except PipelineError as e:
        typer.echo(f"\n✗ Error: {e}", err=True)
        raise typer.Exit(1) from None


@app.command()
def burn(
    video: Annotated[Path, typer.Argument(help="Input video file", exists=True)],
    subs: Annotated[Path, typer.Option("--subs", "-s", help="Subtitle file", exists=True)],
    config: ConfigOption = None,
    out_dir: OutDirOption = None,
    overwrite: OverwriteOption = False,
    log_level: LogLevelOption = "info",
    subtitle_bg: SubtitleBgOption = None,
    preset: PresetOption = None,
) -> None:
    """Burn existing subtitle file into video."""
    _setup_logging(log_level)

    overrides = _build_overrides(
        out_dir=out_dir,
        subtitle_bg=subtitle_bg,
        overwrite=overwrite,
        preset=preset,
    )
    cfg = load_config(config, overrides)

    from vidsub.burn import burn_subtitles

    out_dir_path = Path(cfg.app.out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    output_path = out_dir_path / f"{video.stem}_burned.mp4"

    typer.echo(f"Burning subtitles: {subs} -> {video}")

    try:
        burn_subtitles(video, subs, output_path, cfg, preset_name=preset)
        typer.echo(f"\n✓ Burn complete: {output_path}")
    except Exception as e:
        typer.echo(f"\n✗ Error: {e}", err=True)
        raise typer.Exit(1) from None


@app.command()
def init_config(
    path: Annotated[Path, typer.Option("--path", "-p", help="Config file path")] = Path(
        "vidsub.yaml"
    ),
) -> None:
    """Initialize a configuration file template."""
    if path.exists():
        typer.confirm(f"{path} already exists. Overwrite?", abort=True)
    write_config_template(path)


@app.command()
def validate(
    file: Annotated[Path, typer.Argument(help="File to validate", exists=True)],
) -> None:
    """Validate a transcript JSON or subtitle file."""
    suffix = file.suffix.lower()

    if suffix == ".json":
        typer.echo(f"Validating transcript: {file}")
        from vidsub.models import CanonicalTranscript
        try:
            data = CanonicalTranscript.model_validate_json(file.read_text())
            typer.echo(f"✓ Valid transcript with {len(data.segments)} segments")
        except Exception as e:
            typer.echo(f"✗ Invalid: {e}", err=True)
            raise typer.Exit(1) from None
    elif suffix in (".srt", ".ass"):
        typer.echo(f"Validating subtitles: {file}")
        try:
            if suffix == ".srt":
                from vidsub.subtitles import SrtParser
                srt_parser = SrtParser()
                captions = srt_parser.parse_file(file)
            else:
                from vidsub.subtitles import AssParser
                ass_parser = AssParser()
                captions = ass_parser.parse_file(file)
            typer.echo(f"✓ Valid subtitle file with {len(captions)} captions")
        except Exception as e:
            typer.echo(f"✗ Invalid: {e}", err=True)
            raise typer.Exit(1) from None
    else:
        raise typer.BadParameter(f"Unsupported file format: {suffix}")


def _setup_logging(log_level: str) -> None:
    """Setup logging configuration."""
    import logging

    level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
