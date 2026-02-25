"""FFmpeg and ffprobe wrappers for video processing."""

from __future__ import annotations

import itertools
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VideoInfo:
    """Video file metadata."""

    path: Path
    duration_sec: float
    width: int
    height: int
    fps: float
    video_codec: str
    audio_codec: str | None
    audio_sample_rate: int | None
    has_audio: bool

    @property
    def resolution(self) -> str:
        """Return resolution as WIDTHxHEIGHT."""
        return f"{self.width}x{self.height}"


class FFmpegError(Exception):
    """Base exception for FFmpeg operations."""
    pass


class FFprobeError(FFmpegError):
    """Exception for ffprobe failures."""
    pass


SUPPORTED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
SUPPORTED_VIDEO_CODECS = {"h264", "hevc", "vp8", "vp9", "av1", "mpeg4"}
SUPPORTED_AUDIO_CODECS = {"aac", "mp3", "opus", "vorbis", "flac", "pcm_s16le"}


def find_executable(name: str) -> str | None:
    """Find an executable in PATH."""
    return shutil.which(name)


def get_ffprobe_path() -> str:
    """Get ffprobe path or raise error."""
    path = find_executable("ffprobe")
    if not path:
        raise FFprobeError(
            "ffprobe not found. Please install FFmpeg: https://ffmpeg.org/download.html"
        )
    return path


def get_ffmpeg_path() -> str:
    """Get ffmpeg path or raise error."""
    path = find_executable("ffmpeg")
    if not path:
        raise FFmpegError(
            "ffmpeg not found. Please install FFmpeg: https://ffmpeg.org/download.html"
        )
    return path


def probe_video(video_path: Path) -> VideoInfo:
    """Probe video file for metadata.

    Args:
        video_path: Path to video file.

    Returns:
        VideoInfo with extracted metadata.

    Raises:
        FFprobeError: If probing fails.
        FileNotFoundError: If video file doesn't exist.
    """
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    ffprobe = get_ffprobe_path()

    cmd = [
        ffprobe,
        "-v", "error",
        "-show_entries", "format=duration",
        "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate,sample_rate",
        "-of", "json",
        str(video_path),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        raise FFprobeError(f"ffprobe failed: {e.stderr}") from e

    data = json.loads(result.stdout)
    return _parse_probe_data(video_path, data)


def _parse_probe_data(video_path: Path, data: dict[str, Any]) -> VideoInfo:
    """Parse ffprobe JSON output into VideoInfo."""
    format_info = data.get("format", {})
    streams = data.get("streams", [])

    duration = float(format_info.get("duration", 0))

    # Find video stream
    video_stream = None
    audio_stream = None
    for stream in streams:
        codec_type = stream.get("codec_type")
        if codec_type == "video" and video_stream is None:
            video_stream = stream
        elif codec_type == "audio" and audio_stream is None:
            audio_stream = stream

    if not video_stream:
        raise FFprobeError(f"No video stream found in {video_path}")

    # Parse video stream info
    width = int(video_stream.get("width", 0))
    height = int(video_stream.get("height", 0))
    video_codec = video_stream.get("codec_name", "unknown")

    # Parse frame rate (can be fraction like "30000/1001")
    fps_str = video_stream.get("r_frame_rate", "0")
    fps = _parse_fraction(fps_str)

    # Parse audio info
    audio_codec = None
    audio_sample_rate = None
    has_audio = audio_stream is not None

    if audio_stream:
        audio_codec = audio_stream.get("codec_name")
        sample_rate_str = audio_stream.get("sample_rate")
        if sample_rate_str:
            audio_sample_rate = int(sample_rate_str)

    return VideoInfo(
        path=video_path,
        duration_sec=duration,
        width=width,
        height=height,
        fps=fps,
        video_codec=video_codec,
        audio_codec=audio_codec,
        audio_sample_rate=audio_sample_rate,
        has_audio=has_audio,
    )


def _parse_fraction(fraction_str: str) -> float:
    """Parse fraction string like '30000/1001' to float."""
    if "/" in fraction_str:
        num, denom = fraction_str.split("/")
        if denom == "0":
            return 0.0
        return float(num) / float(denom)
    return float(fraction_str)


def extract_audio(
    video_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    codec: str = "pcm_s16le",
) -> Path:
    """Extract audio from video to WAV file.

    Args:
        video_path: Input video file.
        output_path: Output audio file (typically .wav).
        sample_rate: Target sample rate in Hz (default 16kHz for Whisper).
        channels: Number of audio channels (1=mono, 2=stereo).
        codec: Audio codec (pcm_s16le for 16-bit PCM WAV).

    Returns:
        Path to extracted audio file.

    Raises:
        FFmpegError: If extraction fails.
    """
    ffmpeg = get_ffmpeg_path()

    cmd = [
        ffmpeg,
        "-y",  # Overwrite output
        "-i", str(video_path),
        "-vn",  # No video
        "-acodec", codec,
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-f", "wav",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Audio extraction failed: {e.stderr}") from e

    if not output_path.exists():
        raise FFmpegError(f"Output file not created: {output_path}")

    return output_path


def extract_audio_segment(
    video_path: Path,
    output_path: Path,
    start_sec: float,
    duration_sec: float,
    sample_rate: int = 16000,
) -> Path:
    """Extract a segment of audio from video.

    Used for Gemini chunking strategy.

    Args:
        video_path: Input video file.
        output_path: Output audio file.
        start_sec: Start time in seconds.
        duration_sec: Duration in seconds.
        sample_rate: Target sample rate.

    Returns:
        Path to extracted segment.
    """
    ffmpeg = get_ffmpeg_path()

    cmd = [
        ffmpeg,
        "-y",
        "-ss", str(start_sec),
        "-t", str(duration_sec),
        "-i", str(video_path),
        "-vn",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate),
        "-ac", "1",
        "-f", "wav",
        str(output_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as e:
        raise FFmpegError(f"Segment extraction failed: {e.stderr}") from e

    return output_path


def validate_video_file(video_path: Path) -> tuple[bool, list[str]]:
    """Validate video file for processing.

    Args:
        video_path: Path to video file.

    Returns:
        Tuple of (is_valid, list_of_issues).
    """
    issues: list[str] = []

    # Check existence
    if not video_path.exists():
        return False, [f"File not found: {video_path}"]

    # Check extension
    ext = video_path.suffix.lower()
    if ext not in SUPPORTED_VIDEO_EXTENSIONS:
        issues.append(f"Extension '{ext}' may not be supported (best effort)")

    # Probe file
    try:
        info = probe_video(video_path)
    except FFprobeError as e:
        return False, [f"Cannot probe video: {e}"]

    # Validate duration
    if info.duration_sec <= 0:
        issues.append("Video has no duration")
    elif info.duration_sec > 24 * 60 * 60:  # 24 hours
        issues.append("Video duration exceeds 24 hours")

    # Validate video codec
    if info.video_codec not in SUPPORTED_VIDEO_CODECS:
        issues.append(f"Video codec '{info.video_codec}' may not be supported")

    # Validate has audio (warning only)
    if not info.has_audio:
        issues.append("No audio stream detected")

    # Validate dimensions
    if info.width == 0 or info.height == 0:
        issues.append("Invalid video dimensions")

    is_valid = not any(
        issue.startswith("Cannot") or
        issue.startswith("Video has no") or
        issue.startswith("Invalid")
        for issue in issues
    )

    return is_valid, issues


def check_ffmpeg_capabilities() -> dict[str, Any]:
    """Check available FFmpeg codecs and features.

    Returns:
        Dictionary with capability information.
    """
    ffmpeg = get_ffmpeg_path()

    # Check encoders
    result = subprocess.run(
        [ffmpeg, "-encoders"],
        capture_output=True,
        text=True,
        check=True,
    )
    encoders = result.stdout

    # Check filters (for subtitle burn-in)
    result = subprocess.run(
        [ffmpeg, "-filters"],
        capture_output=True,
        text=True,
        check=True,
    )
    filters = result.stdout

    return {
        "has_h264_encoder": "libx264" in encoders or "h264" in encoders,
        "has_ass_filter": "ass" in filters,
        "has_subtitles_filter": "subtitles" in filters,
        "version": get_ffmpeg_version(),
    }


def get_ffmpeg_version() -> tuple[int, int, int]:
    """Get FFmpeg version as tuple (major, minor, patch)."""
    ffmpeg = get_ffmpeg_path()

    result = subprocess.run(
        [ffmpeg, "-version"],
        capture_output=True,
        text=True,
        check=True,
    )

    # Parse version from first line: "ffmpeg version 6.1.1"
    first_line = result.stdout.splitlines()[0]
    parts = first_line.split()
    for i, part in enumerate(parts):
        if part == "version" and i + 1 < len(parts):
            version_str = parts[i + 1]
            # Remove suffix like "-Copyright" or "-static"
            version_str = version_str.split("-")[0]
            try:
                version_parts = version_str.split(".")
                # Extract digits from each part, handling prefixes like 'n' in 'n8.0.1'
                def extract_digits(s: str) -> str:
                    """Extract leading digits from string, or trailing digits if no leading."""
                    digits = "".join(itertools.takewhile(str.isdigit, s))
                    if digits:
                        return digits
                    # Try trailing digits (e.g., 'n8' -> '8')
                    return "".join(itertools.dropwhile(lambda c: not c.isdigit(), s))

                major_str = extract_digits(version_parts[0])
                major = int(major_str) if major_str else 0
                minor = int(version_parts[1]) if len(version_parts) > 1 and extract_digits(version_parts[1]) else 0
                patch_str = extract_digits(version_parts[2]) if len(version_parts) > 2 else "0"
                patch = int(patch_str) if patch_str else 0
                return (major, minor, patch)
            except (ValueError, IndexError):
                continue

    raise FFmpegError(f"Could not parse FFmpeg version from: {first_line}")


def verify_ffmpeg_installation() -> None:
    """Verify FFmpeg is installed and meets minimum version.

    Raises:
        RuntimeError: If FFmpeg is not found or version is too old.
    """
    ffmpeg = get_ffmpeg_path()
    ffprobe = get_ffprobe_path()

    if not ffmpeg:
        raise RuntimeError(
            "FFmpeg not found. Please install FFmpeg: "
            "https://ffmpeg.org/download.html"
        )

    if not ffprobe:
        raise RuntimeError(
            "ffprobe not found. Please install FFmpeg (includes ffprobe): "
            "https://ffmpeg.org/download.html"
        )

    version = get_ffmpeg_version()
    min_version = (4, 4, 0)
    if version < min_version:
        raise RuntimeError(
            f"FFmpeg version {version} is too old. "
            f"Minimum required: {min_version}"
        )
