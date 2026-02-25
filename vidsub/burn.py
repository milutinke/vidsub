"""Subtitle burn-in rendering using FFmpeg."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

from vidsub.ffmpeg import FFmpegError, get_ffmpeg_path
from vidsub.models import FFmpegPreset

if TYPE_CHECKING:
    from vidsub.models import Config

logger = logging.getLogger(__name__)


class BurnError(FFmpegError):
    """Exception for burn-in failures."""
    pass


def _check_ffmpeg_subtitles_filter(ffmpeg: str) -> None:
    """Check if FFmpeg has the subtitles filter (requires libass).

    Args:
        ffmpeg: Path to FFmpeg binary.

    Raises:
        BurnError: If FFmpeg doesn't have the subtitles filter.
    """
    try:
        result = subprocess.run(
            [ffmpeg, "-filters"],
            capture_output=True,
            text=True,
            check=True,
        )
        if "subtitles" not in result.stdout:
            raise BurnError(
                "FFmpeg does not have the 'subtitles' filter. "
                "This usually means FFmpeg was built without libass support.\n\n"
                "To fix this on macOS, use the ffmpeg-full formula:\n"
                "  brew tap homebrew-ffmpeg/ffmpeg\n"
                "  brew install homebrew-ffmpeg/ffmpeg/ffmpeg\n\n"
                "Or download a FFmpeg build with libass enabled:\n"
                "  https://ffmpeg.org/download.html"
            )
    except subprocess.CalledProcessError as e:
        logger.warning(f"Could not check FFmpeg filters: {e}")


def _get_preset(config: Config, preset_name: str | None = None) -> FFmpegPreset:
    """Get FFmpeg preset to use for encoding.

    Args:
        config: Application configuration.
        preset_name: Optional preset name override.

    Returns:
        FFmpegPreset to use.

    Raises:
        BurnError: If specified preset not found.
    """
    ffmpeg_config = config.ffmpeg

    # Determine preset name to use
    name = preset_name or ffmpeg_config.default_preset

    # Look up preset
    if name in ffmpeg_config.presets:
        preset = ffmpeg_config.presets[name]
        # Ensure name field is set
        if not preset.name:
            preset.name = name
        return preset

    # Preset not found
    available = list(ffmpeg_config.presets.keys())
    raise BurnError(
        f"Unknown FFmpeg preset: '{name}'. "
        f"Available presets: {', '.join(available)}"
    )


def burn_subtitles(
    video_path: Path,
    subtitle_path: Path,
    output_path: Path,
    config: Config,
    preset_name: str | None = None,
) -> Path:
    """Burn subtitles into video.

    Args:
        video_path: Input video file.
        subtitle_path: Subtitle file (SRT or ASS).
        output_path: Output video file.
        config: Application configuration.

    Returns:
        Path to output video.

    Raises:
        BurnError: If burn-in fails.
    """
    ffmpeg = get_ffmpeg_path()

    # Validate inputs first (before checking FFmpeg capabilities)
    if not video_path.exists():
        raise BurnError(f"Video file not found: {video_path}")
    if not subtitle_path.exists():
        raise BurnError(f"Subtitle file not found: {subtitle_path}")

    # Get preset to use
    preset = _get_preset(config, preset_name)

    # Determine subtitle format
    suffix = subtitle_path.suffix.lower()

    if suffix == ".ass":
        # ASS files have styling built-in
        filter_str = f"subtitles={_escape_path(subtitle_path)}"
    elif suffix == ".srt":
        # SRT files - use basic subtitles filter without force_style
        # force_style doesn't work reliably with SRT in FFmpeg
        filter_str = f"subtitles={_escape_path(subtitle_path)}"
    else:
        raise BurnError(f"Unsupported subtitle format: {suffix}")

    # Check FFmpeg has required filters (after validation, before building command)
    _check_ffmpeg_subtitles_filter(ffmpeg)

    # Build FFmpeg command from preset
    cmd = [
        ffmpeg,
        "-y",  # Overwrite output
        "-i", str(video_path),
        "-vf", filter_str,
    ]

    # Audio codec
    cmd.extend(["-c:a", preset.audio_codec])
    if preset.audio_bitrate:
        cmd.extend(["-b:a", preset.audio_bitrate])

    # Copy subtitle streams if present
    cmd.extend(["-c:s", "copy"])

    # Video codec and settings from preset
    cmd.extend(["-c:v", preset.video_codec])
    cmd.extend(["-crf", str(preset.crf)])
    cmd.extend(["-preset", preset.preset])
    if preset.video_bitrate:
        cmd.extend(["-b:v", preset.video_bitrate])
    cmd.extend(["-pix_fmt", preset.pixel_format])

    # Extra args from preset
    if preset.extra_args:
        cmd.extend(preset.extra_args)

    # Output file
    cmd.append(str(output_path))

    logger.info(f"Burning subtitles: {subtitle_path} into {video_path}")
    logger.info(f"Using FFmpeg preset: {preset.name} ({preset.description or 'no description'})")
    logger.debug(f"FFmpeg command: {' '.join(cmd)}")

    try:
        subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise BurnError(f"FFmpeg burn-in failed: {e.stderr}") from e

    if not output_path.exists():
        raise BurnError(f"Output file not created: {output_path}")

    # Verify output duration matches input
    _verify_output(video_path, output_path)

    logger.info(f"Burn-in complete: {output_path}")
    return output_path


def _escape_path(path: Path) -> str:
    """Escape path for FFmpeg filter syntax.

    Args:
        path: Path to escape.

    Returns:
        Escaped path string.
    """
    # In filter expressions, ':' and '\' need escaping
    path_str = str(path.absolute())
    path_str = path_str.replace("\\", "/")  # Use forward slashes
    path_str = path_str.replace(":", "\\:")  # Escape colons
    return path_str


def _build_srt_filter(subtitle_path: Path, config: Config) -> str:
    """Build FFmpeg filter string for SRT subtitles.

    Args:
        subtitle_path: Path to SRT file.
        config: Application configuration.

    Returns:
        FFmpeg filter string.
    """
    style = config.style_ass

    # Build subtitle filter with font options
    # Note: FFmpeg's subtitles filter with SRT uses force_style
    # The entire filter must be quoted to protect special characters
    escaped_path = _escape_path(subtitle_path)
    filter_str = f"subtitles={escaped_path}:force_style='FontName={style.font_name},FontSize={style.font_size}'"

    return filter_str


def _verify_output(input_path: Path, output_path: Path) -> None:
    """Verify output video is valid and duration matches.

    Args:
        input_path: Original input video.
        output_path: Burned output video.

    Raises:
        BurnError: If verification fails.
    """
    from vidsub.ffmpeg import probe_video

    try:
        input_info = probe_video(input_path)
        output_info = probe_video(output_path)
    except Exception as e:
        raise BurnError(f"Failed to verify output: {e}") from e

    # Check duration within 250ms tolerance (per PRD)
    duration_diff = abs(input_info.duration_sec - output_info.duration_sec)
    if duration_diff > 0.25:
        raise BurnError(
            f"Output duration mismatch: input={input_info.duration_sec:.3f}s, "
            f"output={output_info.duration_sec:.3f}s, "
            f"diff={duration_diff:.3f}s (max 0.25s)"
        )

    # Check video stream exists
    if output_info.width == 0 or output_info.height == 0:
        raise BurnError("Output video has invalid dimensions")

    logger.debug(
        f"Verification passed: duration_diff={duration_diff:.3f}s"
    )


def preview_subtitles(
    video_path: Path,
    subtitle_path: Path,
    output_path: Path,
    start_sec: float = 0.0,
    duration_sec: float = 10.0,
    config: Config | None = None,
    preset_name: str | None = None,
) -> Path:
    """Create a preview clip with burned subtitles.

    Useful for testing subtitle styling without processing entire video.

    Args:
        video_path: Input video file.
        subtitle_path: Subtitle file.
        output_path: Output preview file.
        start_sec: Start time for preview.
        duration_sec: Duration of preview.
        config: Optional configuration.
        preset_name: Optional FFmpeg preset name.

    Returns:
        Path to preview video.
    """
    ffmpeg = get_ffmpeg_path()

    # Get preset (use default or specified)
    if config:
        preset = _get_preset(config, preset_name)
    else:
        # Fallback to hardcoded defaults for preview
        preset = FFmpegPreset(name="preview", crf=23, preset="fast")

    suffix = subtitle_path.suffix.lower()
    if suffix == ".ass":
        filter_str = f"subtitles={_escape_path(subtitle_path)}"
    else:
        filter_str = _build_srt_filter(subtitle_path, config) if config else f"subtitles={_escape_path(subtitle_path)}"

    cmd = [
        ffmpeg,
        "-y",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", str(video_path),
        "-vf", filter_str,
        "-c:a", preset.audio_codec,
        "-c:v", preset.video_codec,
        "-crf", str(preset.crf),
        "-preset", preset.preset,
        "-pix_fmt", preset.pixel_format,
    ]

    # Add extra args if present
    if preset.extra_args:
        cmd.extend(preset.extra_args)

    cmd.append(str(output_path))

    logger.info(f"Creating preview: {start_sec}s to {start_sec + duration_sec}s")
    logger.debug(f"Using preset: {preset.name}")

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise BurnError(f"Preview creation failed: {e.stderr}") from e

    return output_path
