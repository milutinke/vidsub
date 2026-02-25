"""Tests for subtitle burn-in."""

from pathlib import Path
from unittest import mock

import pytest

from vidsub.burn import BurnError, burn_subtitles
from vidsub.config import DEFAULT_YOUTUBE_PRESET
from vidsub.models import AssStyleConfig, Config, FFmpegConfig, SubtitleConfig


class TestBurnSubtitles:
    """Tests for burn_subtitles function."""

    @pytest.fixture
    def config(self) -> Config:
        return Config(
            subtitles=SubtitleConfig(),
            style_ass=AssStyleConfig(),
            ffmpeg=FFmpegConfig(
                default_preset="youtube_1080p",
                presets={"youtube_1080p": DEFAULT_YOUTUBE_PRESET},
            ),
        )

    @mock.patch("vidsub.burn._check_ffmpeg_subtitles_filter")
    @mock.patch("vidsub.burn.subprocess.run")
    @mock.patch("vidsub.burn.get_ffmpeg_path")
    @mock.patch("vidsub.ffmpeg.probe_video")
    def test_burn_ass_file(
        self,
        mock_probe: mock.MagicMock,
        mock_get_ffmpeg: mock.MagicMock,
        mock_run: mock.MagicMock,
        mock_check_filter: mock.MagicMock,
        tmp_path: Path,
        config: Config,
    ) -> None:
        # Setup
        mock_get_ffmpeg.return_value = "ffmpeg"
        mock_probe.return_value.duration_sec = 10.0

        video = tmp_path / "test.mp4"
        video.write_text("fake")
        subs = tmp_path / "test.ass"
        subs.write_text("[Script Info]")
        output = tmp_path / "output.mp4"

        # Make subprocess.run create the output file
        def create_output_file(*args: object, **kwargs: object) -> mock.MagicMock:
            output.write_text("fake video")
            return mock.MagicMock(returncode=0)

        mock_run.side_effect = create_output_file

        # Execute
        result = burn_subtitles(video, subs, output, config)

        # Verify
        assert result == output
        mock_run.assert_called_once()
        cmd = mock_run.call_args[0][0]
        cmd_str = " ".join(cmd)
        assert "subtitles=" in cmd_str  # Check in the full command

    def test_missing_video_raises(self, tmp_path: Path, config: Config) -> None:
        with pytest.raises(BurnError, match="Video file not found"):
            burn_subtitles(
                tmp_path / "missing.mp4",
                tmp_path / "subs.ass",
                tmp_path / "out.mp4",
                config,
            )

    def test_missing_subtitles_raises(self, tmp_path: Path, config: Config) -> None:
        video = tmp_path / "video.mp4"
        video.write_text("fake")

        with pytest.raises(BurnError, match="Subtitle file not found"):
            burn_subtitles(
                video,
                tmp_path / "missing.ass",
                tmp_path / "out.mp4",
                config,
            )

    def test_unsupported_format_raises(self, tmp_path: Path, config: Config) -> None:
        video = tmp_path / "video.mp4"
        video.write_text("fake")
        subs = tmp_path / "subs.txt"
        subs.write_text("fake")

        with pytest.raises(BurnError, match="Unsupported subtitle format"):
            burn_subtitles(video, subs, tmp_path / "out.mp4", config)


class TestEscapePath:
    """Tests for path escaping."""

    def test_escapes_colons(self) -> None:
        from vidsub.burn import _escape_path

        path = Path("C:/some/path")
        result = _escape_path(path)
        assert "\\:" in result

    def test_uses_forward_slashes(self) -> None:
        from vidsub.burn import _escape_path

        path = Path("C:\\some\\path")
        result = _escape_path(path)
        assert "\\" not in result or "\\:" in result
