"""Tests for FFmpeg wrappers."""

from pathlib import Path

import pytest

from vidsub.ffmpeg import (
    FFprobeError,
    VideoInfo,
    check_ffmpeg_capabilities,
    extract_audio,
    extract_audio_segment,
    get_ffmpeg_version,
    probe_video,
    validate_video_file,
)


class TestProbeVideo:
    """Tests for video probing."""

    def test_probe_video_success(self, sample_video: Path) -> None:
        """Test successful video probing."""
        info = probe_video(sample_video)

        assert isinstance(info, VideoInfo)
        assert info.path == sample_video
        assert info.duration_sec > 0
        assert info.width > 0
        assert info.height > 0
        assert info.video_codec
        assert info.has_audio is True

    def test_probe_nonexistent_file(self) -> None:
        """Test probing non-existent file raises."""
        with pytest.raises(FileNotFoundError):
            probe_video(Path("/nonexistent/video.mp4"))

    def test_probe_invalid_file(self, tmp_path: Path) -> None:
        """Test probing invalid file raises."""
        invalid_file = tmp_path / "invalid.mp4"
        invalid_file.write_text("not a video")

        with pytest.raises(FFprobeError):
            probe_video(invalid_file)


class TestExtractAudio:
    """Tests for audio extraction."""

    def test_extract_audio(self, sample_video: Path, tmp_path: Path) -> None:
        """Test extracting audio from video."""
        output = tmp_path / "output.wav"
        result = extract_audio(sample_video, output)

        assert result == output
        assert output.exists()
        assert output.stat().st_size > 0

    def test_extract_audio_segment(self, sample_video: Path, tmp_path: Path) -> None:
        """Test extracting audio segment."""
        output = tmp_path / "segment.wav"
        result = extract_audio_segment(
            sample_video, output, start_sec=0.0, duration_sec=1.0
        )

        assert result == output
        assert output.exists()


class TestValidateVideoFile:
    """Tests for video validation."""

    def test_valid_video(self, sample_video: Path) -> None:
        """Test valid video passes validation."""
        is_valid, issues = validate_video_file(sample_video)
        assert is_valid is True
        assert issues == [] or all("may not be" in i for i in issues)

    def test_nonexistent_file(self) -> None:
        """Test non-existent file fails validation."""
        is_valid, issues = validate_video_file(Path("/nonexistent.mp4"))
        assert is_valid is False
        assert any("not found" in i for i in issues)


class TestGetFfmpegVersion:
    """Tests for version detection."""

    def test_get_version(self) -> None:
        """Test version detection returns tuple."""
        version = get_ffmpeg_version()
        assert isinstance(version, tuple)
        assert len(version) == 3
        assert all(isinstance(v, int) for v in version)
        assert version[0] >= 4  # At least FFmpeg 4.x


class TestCheckCapabilities:
    """Tests for capability checking."""

    def test_capabilities(self) -> None:
        """Test capability check returns expected fields."""
        caps = check_ffmpeg_capabilities()

        assert "has_h264_encoder" in caps
        assert "has_ass_filter" in caps
        assert "has_subtitles_filter" in caps
        assert "version" in caps
