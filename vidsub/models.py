"""Pydantic models for vidsub data structures."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


class Word(BaseModel):
    """A single word with timing and confidence."""

    start: float = Field(ge=0, description="Start time in seconds")
    end: float = Field(ge=0, description="End time in seconds")
    word: str = Field(min_length=1, description="The word text")
    confidence: float | None = Field(
        default=None, ge=0, le=1, description="Optional confidence score"
    )

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info: Any) -> float:
        """Ensure end >= start."""
        if info.data.get("start") is not None and v < info.data["start"]:
            raise ValueError("end must be >= start")
        return v


class Segment(BaseModel):
    """A transcript segment (caption)."""

    start: float = Field(ge=0, description="Start time in seconds")
    end: float = Field(ge=0, description="End time in seconds")
    text: str = Field(min_length=1, description="Segment text")
    speaker: str | None = Field(default=None, description="Optional speaker ID")
    words: list[Word] | None = Field(default=None, description="Optional word-level timing")

    @field_validator("end")
    @classmethod
    def end_after_start(cls, v: float, info: Any) -> float:
        """Ensure end >= start."""
        if info.data.get("start") is not None and v < info.data["start"]:
            raise ValueError("end must be >= start")
        return v


class CanonicalTranscript(BaseModel):
    """Canonical transcript format - internal representation.

    This is the interchange format between transcription engines
    and subtitle generators.
    """

    engine: Literal["whisper", "gemini"] = Field(description="Source engine")
    model: str = Field(description="Model identifier used")
    language: str = Field(description="ISO 639-1 language code")
    duration_sec: float = Field(gt=0, description="Total media duration in seconds")
    segments: list[Segment] = Field(description="Transcript segments")


class AppConfig(BaseModel):
    """Application-level configuration."""

    out_dir: str = "./out"
    temp_dir: str = "./.vidsub_tmp"
    overwrite: bool = False
    keep_temp: bool = False
    burn: bool = False
    verbose_postprocessing: bool = False
    show_progress: bool = True


class WhisperConfig(BaseModel):
    """Whisper engine configuration."""

    model: Literal["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"] = "large"
    device: Literal["auto", "cpu", "cuda", "mps"] = "auto"
    vad: bool = True
    accurate: bool = True


class GeminiConfig(BaseModel):
    """Gemini engine configuration."""

    model: str = "gemini-2.5-pro"
    api_key_env: str = "GEMINI_API_KEY"
    chunk_seconds: int = Field(default=60, ge=10, le=180)
    overlap_seconds: int = Field(default=2, ge=0, le=10)
    fps: int = Field(default=1, ge=1, le=10)
    max_retries: int = Field(default=2, ge=0, le=5)
    concurrency: int = Field(default=3, ge=1, le=8)
    upload_timeout_sec: int = Field(default=180, ge=1, le=600)
    poll_interval_sec: float = Field(default=2.0, gt=0, le=30.0)
    retry_base_delay_sec: float = Field(default=1.0, gt=0, le=30.0)
    retry_max_delay_sec: float = Field(default=8.0, gt=0, le=120.0)


class SubtitleConfig(BaseModel):
    """Subtitle generation configuration."""

    formats: list[Literal["srt", "ass"]] = ["srt", "ass"]
    max_chars_per_line: int = Field(default=42, ge=20, le=100)
    max_lines: int = Field(default=2, ge=1, le=3)
    max_caption_seconds: float = Field(default=6.0, ge=1.0, le=20.0)
    split_on_silence_ms: int = Field(default=350, ge=0, le=1000)
    background_style: Literal["none", "solid"] = "none"


class AssStyleConfig(BaseModel):
    """ASS subtitle styling configuration."""

    font_name: str = "Inter"
    font_size: int = Field(default=44, ge=10, le=200)
    outline: int = Field(default=3, ge=0, le=10)
    shadow: int = Field(default=0, ge=0, le=10)
    margin_v: int = Field(default=40, ge=0, le=500)
    bg_color: str = "black"


class EngineConfig(BaseModel):
    """Engine selection configuration."""

    name: Literal["whisper", "gemini"] = "whisper"
    language: str = "en"


class FFmpegPreset(BaseModel):
    """FFmpeg encoding preset configuration."""

    name: str = Field(default="", description="Preset identifier (set from dict key)")
    video_codec: str = Field(default="libx264", description="Video codec (e.g., libx264, libx265)")
    audio_codec: str = Field(default="copy", description="Audio codec or 'copy' to pass through")
    crf: int = Field(default=18, ge=0, le=51, description="Constant rate factor (0-51, lower is better)")
    preset: str = Field(default="medium", description="Encoding speed preset (ultrafast to veryslow)")
    pixel_format: str = Field(default="yuv420p", description="Pixel format for compatibility")
    video_bitrate: str | None = Field(default=None, description="Optional video bitrate (e.g., '5M')")
    audio_bitrate: str | None = Field(default=None, description="Optional audio bitrate (e.g., '192k')")
    extra_args: list[str] = Field(default_factory=list, description="Additional FFmpeg arguments")
    description: str = Field(default="", description="Human-readable description")


class FFmpegConfig(BaseModel):
    """FFmpeg configuration section."""

    presets: dict[str, FFmpegPreset] = Field(
        default_factory=dict,
        description="Named FFmpeg encoding presets"
    )
    default_preset: str = Field(
        default="youtube_1080p",
        description="Default preset to use when none specified"
    )


class Config(BaseModel):
    """Root configuration object."""

    app: AppConfig = Field(default_factory=AppConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    gemini: GeminiConfig = Field(default_factory=GeminiConfig)
    subtitles: SubtitleConfig = Field(default_factory=SubtitleConfig)
    style_ass: AssStyleConfig = Field(default_factory=AssStyleConfig)
    ffmpeg: FFmpegConfig = Field(default_factory=FFmpegConfig)
