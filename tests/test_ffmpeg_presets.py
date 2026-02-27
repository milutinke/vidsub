"""Tests for FFmpeg preset configuration."""

from pathlib import Path
from typing import Any
from unittest import mock

import pytest

from vidsub.burn import BurnError, _get_preset, burn_subtitles
from vidsub.config import DEFAULT_YOUTUBE_PRESET, _apply_default_ffmpeg_presets
from vidsub.models import Config, FFmpegConfig, FFmpegPreset


class TestFFmpegPresetModel:
    """Tests for FFmpegPreset model."""

    def test_default_values(self) -> None:
        """Test preset has correct defaults."""
        preset = FFmpegPreset(name="test")

        assert preset.video_codec == "libx264"
        assert preset.audio_codec == "copy"
        assert preset.crf == 18
        assert preset.preset == "medium"
        assert preset.pixel_format == "yuv420p"
        assert preset.video_bitrate is None
        assert preset.audio_bitrate is None
        assert preset.extra_args == []
        assert preset.description == ""

    def test_custom_values(self) -> None:
        """Test preset accepts custom values."""
        preset = FFmpegPreset(
            name="custom",
            video_codec="libx265",
            audio_codec="aac",
            crf=23,
            preset="slow",
            pixel_format="yuv444p",
            video_bitrate="5M",
            audio_bitrate="192k",
            extra_args=["-movflags", "+faststart"],
            description="Custom preset for testing",
        )

        assert preset.video_codec == "libx265"
        assert preset.audio_codec == "aac"
        assert preset.crf == 23
        assert preset.preset == "slow"
        assert preset.pixel_format == "yuv444p"
        assert preset.video_bitrate == "5M"
        assert preset.audio_bitrate == "192k"
        assert preset.extra_args == ["-movflags", "+faststart"]
        assert preset.description == "Custom preset for testing"

    def test_crf_validation(self) -> None:
        """Test CRF value validation."""
        # Valid CRF values
        FFmpegPreset(name="test", crf=0)
        FFmpegPreset(name="test", crf=51)
        FFmpegPreset(name="test", crf=23)

        # Invalid CRF values
        with pytest.raises(ValueError):
            FFmpegPreset(name="test", crf=-1)

        with pytest.raises(ValueError):
            FFmpegPreset(name="test", crf=52)


class TestDefaultYouTubePreset:
    """Tests for default YouTube preset."""

    def test_default_youtube_preset_values(self) -> None:
        """Test the hardcoded YouTube preset has correct values."""
        preset = DEFAULT_YOUTUBE_PRESET

        assert preset.name == "youtube_1080p"
        assert preset.video_codec == "libx264"
        assert preset.audio_codec == "copy"
        assert preset.crf == 18
        assert preset.preset == "medium"
        assert preset.pixel_format == "yuv420p"
        assert preset.description == "Default YouTube 1080p optimized preset"


class TestApplyDefaultFFmpegPresets:
    """Tests for _apply_default_ffmpeg_presets function."""

    def test_empty_config_gets_default_preset(self) -> None:
        """Test that empty config gets default YouTube preset."""
        config_dict: dict[str, Any] = {}

        result = _apply_default_ffmpeg_presets(config_dict)

        assert "ffmpeg" in result
        assert result["ffmpeg"]["default_preset"] == "youtube_1080p"
        assert "youtube_1080p" in result["ffmpeg"]["presets"]

    def test_existing_presets_preserved(self) -> None:
        """Test that existing presets are preserved."""
        config_dict = {
            "ffmpeg": {
                "presets": {
                    "custom": {
                        "name": "custom",
                        "video_codec": "libx265",
                        "crf": 20,
                    }
                }
            }
        }

        result = _apply_default_ffmpeg_presets(config_dict)

        assert "custom" in result["ffmpeg"]["presets"]
        assert result["ffmpeg"]["default_preset"] == "custom"  # First preset becomes default

    def test_default_preset_set_when_missing(self) -> None:
        """Test default_preset is set to first preset if not specified."""
        config_dict = {
            "ffmpeg": {
                "presets": {
                    "first": {"name": "first", "crf": 20},
                    "second": {"name": "second", "crf": 22},
                }
            }
        }

        result = _apply_default_ffmpeg_presets(config_dict)

        assert result["ffmpeg"]["default_preset"] == "first"

    def test_default_preset_preserved_when_specified(self) -> None:
        """Test default_preset is preserved when explicitly specified."""
        config_dict = {
            "ffmpeg": {
                "default_preset": "second",
                "presets": {
                    "first": {"name": "first", "crf": 20},
                    "second": {"name": "second", "crf": 22},
                }
            }
        }

        result = _apply_default_ffmpeg_presets(config_dict)

        assert result["ffmpeg"]["default_preset"] == "second"


class TestGetPreset:
    """Tests for _get_preset function."""

    def test_get_preset_by_name(self) -> None:
        """Test getting preset by name."""
        custom_preset = FFmpegPreset(name="custom", crf=25)
        config = Config(
            ffmpeg=FFmpegConfig(
                default_preset="default",
                presets={
                    "default": DEFAULT_YOUTUBE_PRESET,
                    "custom": custom_preset,
                }
            )
        )

        result = _get_preset(config, "custom")

        assert result.name == "custom"
        assert result.crf == 25

    def test_get_default_preset_when_no_name_specified(self) -> None:
        """Test getting default preset when no name specified."""
        config = Config(
            ffmpeg=FFmpegConfig(
                default_preset="youtube_1080p",
                presets={"youtube_1080p": DEFAULT_YOUTUBE_PRESET}
            )
        )

        result = _get_preset(config, None)

        assert result.name == "youtube_1080p"

    def test_unknown_preset_raises_error(self) -> None:
        """Test that unknown preset raises BurnError."""
        config = Config(
            ffmpeg=FFmpegConfig(
                default_preset="existing",
                presets={"existing": DEFAULT_YOUTUBE_PRESET}
            )
        )

        with pytest.raises(BurnError, match="Unknown FFmpeg preset: 'unknown'"):
            _get_preset(config, "unknown")

    def test_error_includes_available_presets(self) -> None:
        """Test error message includes available presets."""
        config = Config(
            ffmpeg=FFmpegConfig(
                default_preset="preset1",
                presets={
                    "preset1": DEFAULT_YOUTUBE_PRESET,
                    "preset2": FFmpegPreset(name="preset2"),
                }
            )
        )

        with pytest.raises(BurnError, match="preset1"):
            _get_preset(config, "unknown")

        with pytest.raises(BurnError, match="preset2"):
            _get_preset(config, "unknown")


class TestBurnWithPresets:
    """Tests for burn_subtitles with presets."""

    @pytest.fixture
    def config_with_preset(self) -> Config:
        """Create config with a test preset."""
        return Config(
            ffmpeg=FFmpegConfig(
                default_preset="test",
                presets={
                    "test": FFmpegPreset(
                        name="test",
                        video_codec="libx264",
                        audio_codec="copy",
                        crf=20,
                        preset="fast",
                    )
                }
            )
        )

    @mock.patch("vidsub.burn.subprocess.run")
    @mock.patch("vidsub.burn.get_ffmpeg_path")
    @mock.patch("vidsub.burn._check_ffmpeg_subtitles_filter")
    @mock.patch("vidsub.ffmpeg.probe_video")
    def test_burn_uses_preset_values(
        self,
        mock_probe: mock.MagicMock,
        _mock_check_filter: mock.MagicMock,
        mock_get_ffmpeg: mock.MagicMock,
        mock_run: mock.MagicMock,
        tmp_path: Path,
        config_with_preset: Config,
    ) -> None:
        """Test that burn uses values from preset."""
        mock_get_ffmpeg.return_value = "ffmpeg"
        mock_probe.return_value.duration_sec = 10.0

        video = tmp_path / "test.mp4"
        video.write_text("fake")
        subs = tmp_path / "test.ass"
        subs.write_text("[Script Info]")
        output = tmp_path / "output.mp4"

        def create_output(*args: object, **kwargs: object) -> mock.MagicMock:
            output.write_text("fake")
            return mock.MagicMock(returncode=0)

        mock_run.side_effect = create_output

        burn_subtitles(video, subs, output, config_with_preset)

        cmd = mock_run.call_args[0][0]
        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "-crf" in cmd
        assert "20" in cmd
        assert "-preset" in cmd
        assert "fast" in cmd

    @mock.patch("vidsub.burn.subprocess.run")
    @mock.patch("vidsub.burn.get_ffmpeg_path")
    @mock.patch("vidsub.burn._check_ffmpeg_subtitles_filter")
    @mock.patch("vidsub.ffmpeg.probe_video")
    def test_burn_with_custom_preset_name(
        self,
        mock_probe: mock.MagicMock,
        _mock_check_filter: mock.MagicMock,
        mock_get_ffmpeg: mock.MagicMock,
        mock_run: mock.MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test specifying preset name overrides default."""
        mock_get_ffmpeg.return_value = "ffmpeg"
        mock_probe.return_value.duration_sec = 10.0

        config = Config(
            ffmpeg=FFmpegConfig(
                default_preset="default",
                presets={
                    "default": FFmpegPreset(name="default", crf=18),
                    "custom": FFmpegPreset(name="custom", crf=30),
                }
            )
        )

        video = tmp_path / "test.mp4"
        video.write_text("fake")
        subs = tmp_path / "test.ass"
        subs.write_text("[Script Info]")
        output = tmp_path / "output.mp4"

        def create_output(*args: object, **kwargs: object) -> mock.MagicMock:
            output.write_text("fake")
            return mock.MagicMock(returncode=0)

        mock_run.side_effect = create_output

        burn_subtitles(video, subs, output, config, preset_name="custom")

        cmd = mock_run.call_args[0][0]
        assert "30" in cmd  # Custom preset CRF

    @mock.patch("vidsub.burn.subprocess.run")
    @mock.patch("vidsub.burn.get_ffmpeg_path")
    @mock.patch("vidsub.burn._check_ffmpeg_subtitles_filter")
    @mock.patch("vidsub.ffmpeg.probe_video")
    def test_burn_with_audio_bitrate(
        self,
        mock_probe: mock.MagicMock,
        _mock_check_filter: mock.MagicMock,
        mock_get_ffmpeg: mock.MagicMock,
        mock_run: mock.MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test preset with audio bitrate is applied."""
        mock_get_ffmpeg.return_value = "ffmpeg"
        mock_probe.return_value.duration_sec = 10.0

        config = Config(
            ffmpeg=FFmpegConfig(
                default_preset="aac",
                presets={
                    "aac": FFmpegPreset(
                        name="aac",
                        audio_codec="aac",
                        audio_bitrate="192k",
                    )
                }
            )
        )

        video = tmp_path / "test.mp4"
        video.write_text("fake")
        subs = tmp_path / "test.ass"
        subs.write_text("[Script Info]")
        output = tmp_path / "output.mp4"

        def create_output(*args: object, **kwargs: object) -> mock.MagicMock:
            output.write_text("fake")
            return mock.MagicMock(returncode=0)

        mock_run.side_effect = create_output

        burn_subtitles(video, subs, output, config)

        cmd = mock_run.call_args[0][0]
        assert "-c:a" in cmd
        assert "aac" in cmd
        assert "-b:a" in cmd
        assert "192k" in cmd

    @mock.patch("vidsub.burn.subprocess.run")
    @mock.patch("vidsub.burn.get_ffmpeg_path")
    @mock.patch("vidsub.burn._check_ffmpeg_subtitles_filter")
    @mock.patch("vidsub.ffmpeg.probe_video")
    def test_burn_with_extra_args(
        self,
        mock_probe: mock.MagicMock,
        _mock_check_filter: mock.MagicMock,
        mock_get_ffmpeg: mock.MagicMock,
        mock_run: mock.MagicMock,
        tmp_path: Path,
    ) -> None:
        """Test preset with extra_args is applied."""
        mock_get_ffmpeg.return_value = "ffmpeg"
        mock_probe.return_value.duration_sec = 10.0

        config = Config(
            ffmpeg=FFmpegConfig(
                default_preset="extra",
                presets={
                    "extra": FFmpegPreset(
                        name="extra",
                        extra_args=["-movflags", "+faststart"],
                    )
                }
            )
        )

        video = tmp_path / "test.mp4"
        video.write_text("fake")
        subs = tmp_path / "test.ass"
        subs.write_text("[Script Info]")
        output = tmp_path / "output.mp4"

        def create_output(*args: object, **kwargs: object) -> mock.MagicMock:
            output.write_text("fake")
            return mock.MagicMock(returncode=0)

        mock_run.side_effect = create_output

        burn_subtitles(video, subs, output, config)

        cmd = mock_run.call_args[0][0]
        assert "-movflags" in cmd
        assert "+faststart" in cmd
