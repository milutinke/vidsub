"""Configuration loading and management."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from vidsub.models import Config, FFmpegPreset

DEFAULT_YOUTUBE_PRESET = FFmpegPreset(
    name="youtube_1080p",
    video_codec="libx264",
    audio_codec="copy",
    crf=18,
    preset="medium",
    pixel_format="yuv420p",
    description="Default YouTube 1080p optimized preset"
)

DEFAULT_CONFIG_PATHS = [
    Path("vidsub.yaml"),
    Path("vidsub.yml"),
    Path("config/vidsub.yaml"),
    Path(".vidsub/config.yaml"),
]


def find_config_file(path: Path | None = None) -> Path | None:
    """Find configuration file.

    Args:
        path: Explicit config path, or None to search defaults.

    Returns:
        Path to config file, or None if not found.
    """
    if path:
        if path.exists():
            return path
        raise FileNotFoundError(f"Config file not found: {path}")

    for default_path in DEFAULT_CONFIG_PATHS:
        if default_path.exists():
            return default_path

    return None


def load_yaml_config(path: Path) -> dict[str, Any]:
    """Load and parse YAML config file.

    Args:
        path: Path to YAML file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        ValueError: If YAML is invalid.
    """
    try:
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {path}: {e}") from e


def load_config(
    config_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> Config:
    """Load configuration with proper precedence.

    Precedence (highest to lowest):
    1. CLI overrides
    2. Environment variables (for secrets)
    3. YAML config file
    4. Built-in defaults

    Args:
        config_path: Explicit path to config file, or None to search.
        cli_overrides: Dictionary of CLI-provided overrides.

    Returns:
        Fully resolved Config object.
    """
    # Start with defaults
    config_dict: dict[str, Any] = {}

    # Layer 1: YAML config file
    found_path = find_config_file(config_path)
    if found_path:
        config_dict = load_yaml_config(found_path)

    # Layer 2: Environment variables
    # Only for specific sensitive/configurable values
    if os.environ.get("VIDSUB_OUT_DIR"):
        config_dict.setdefault("app", {})["out_dir"] = os.environ["VIDSUB_OUT_DIR"]
    if os.environ.get("VIDSUB_ENGINE"):
        config_dict.setdefault("engine", {})["name"] = os.environ["VIDSUB_ENGINE"]

    # Layer 3: CLI overrides (highest priority)
    if cli_overrides:
        _deep_update(config_dict, cli_overrides)

    # Apply default FFmpeg presets if none defined
    config_dict = _apply_default_ffmpeg_presets(config_dict)

    return Config.model_validate(config_dict)


def _apply_default_ffmpeg_presets(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Apply default YouTube preset if no FFmpeg presets defined.

    Args:
        config_dict: Configuration dictionary to update.

    Returns:
        Updated configuration dictionary.
    """
    ffmpeg_config = config_dict.get("ffmpeg", {})
    presets = ffmpeg_config.get("presets", {})

    if not presets:
        # No presets defined, add default YouTube preset
        config_dict.setdefault("ffmpeg", {})["presets"] = {
            "youtube_1080p": DEFAULT_YOUTUBE_PRESET.model_dump(exclude={"name"})
        }
        # Set default_preset if not already set
        if not ffmpeg_config.get("default_preset"):
            config_dict["ffmpeg"]["default_preset"] = "youtube_1080p"
    else:
        # Presets exist, ensure default_preset is set
        default_preset = ffmpeg_config.get("default_preset")
        if not default_preset:
            # Use first preset as default
            config_dict["ffmpeg"]["default_preset"] = list(presets.keys())[0]

    return config_dict


def _deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep update dictionary, merging nested dicts."""
    for key, value in overrides.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def write_config_template(path: Path) -> None:
    """Write default configuration template to file.

    Args:
        path: Destination path for template.
    """
    template_path = Path(__file__).parent.parent / "config_template.yaml"
    content = (
        template_path.read_text()
        if template_path.exists()
        else _DEFAULT_TEMPLATE  # Fallback template if file not found
    )

    path.write_text(content)
    print(f"Configuration template written to: {path}")


_DEFAULT_TEMPLATE = """# vidsub configuration template
# Copy this file to vidsub.yaml and customize

app:
  out_dir: "./out"
  temp_dir: "./.vidsub_tmp"
  overwrite: false
  keep_temp: false

engine:
  name: "whisper"  # whisper | gemini
  language: "en"

whisper:
  model: "large"
  device: "auto"   # auto, cpu, cuda, mps
  vad: true
  accurate: true

gemini:
  model: "gemini-2.5-pro"
  api_key_env: "GEMINI_API_KEY"
  chunk_seconds: 60
  overlap_seconds: 2
  fps: 1
  max_retries: 2
  concurrency: 3
  upload_timeout_sec: 180
  poll_interval_sec: 2.0
  retry_base_delay_sec: 1.0
  retry_max_delay_sec: 8.0

subtitles:
  formats: ["srt", "ass"]
  max_chars_per_line: 42
  max_lines: 2
  max_caption_seconds: 6.0
  split_on_silence_ms: 350
  background_style: "none"  # none | solid

style_ass:
  font_name: "Inter"
  font_size: 44
  outline: 3
  shadow: 0
  margin_v: 40
  bg_color: "black"

ffmpeg:
  default_preset: "youtube_1080p"
  presets:
    youtube_1080p:
      video_codec: "libx264"
      audio_codec: "copy"
      crf: 18
      preset: "medium"
      pixel_format: "yuv420p"
      description: "Optimized for YouTube 1080p uploads"

    youtube_4k:
      video_codec: "libx264"
      audio_codec: "copy"
      crf: 17
      preset: "slow"
      pixel_format: "yuv420p"
      description: "High quality for YouTube 4K uploads"

    web_optimized:
      video_codec: "libx264"
      audio_codec: "aac"
      audio_bitrate: "128k"
      crf: 23
      preset: "fast"
      pixel_format: "yuv420p"
      description: "Smaller file size for web sharing"

    archive:
      video_codec: "libx264"
      audio_codec: "flac"
      crf: 15
      preset: "veryslow"
      pixel_format: "yuv420p"
      description: "Maximum quality for archival"
"""
