"""Integration tests for complete vidsub workflow."""

import subprocess
from pathlib import Path

import pytest

from vidsub.ffmpeg import probe_video


@pytest.fixture(scope="module")
def sample_video() -> Path:
    """Create a sample video for integration tests."""
    # This assumes a sample video exists or will be created
    test_dir = Path(__file__).parent / "fixtures"
    test_dir.mkdir(exist_ok=True)

    video_path = test_dir / "sample_video.mp4"

    if not video_path.exists():
        # Create a 5-second test video with audio
        try:
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-f", "lavfi",
                    "-i", "testsrc=duration=5:size=640x480:rate=30",
                    "-f", "lavfi",
                    "-i", "sine=frequency=1000:duration=5",
                    "-pix_fmt", "yuv420p",
                    str(video_path),
                ],
                capture_output=True,
                check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("FFmpeg not available for creating sample video")

    return video_path


@pytest.mark.skipif(
    not Path("/usr/bin/ffmpeg").exists() and not Path("/usr/local/bin/ffmpeg").exists(),
    reason="FFmpeg not installed",
)
class TestVideoProcessing:
    """Integration tests for video processing workflow."""

    def test_probe_video(self, sample_video: Path) -> None:
        """Test video probing returns valid metadata."""
        info = probe_video(sample_video)

        assert info.duration_sec > 0
        assert info.width > 0
        assert info.height > 0
        assert info.has_audio

    def test_transcribe_whisper(self, sample_video: Path, tmp_path: Path) -> None:
        """Test Whisper transcription produces valid transcript."""
        pytest.importorskip("whisper_timestamped")

        from vidsub.engines.whisper_engine import WhisperEngine
        from vidsub.models import Config, EngineConfig, WhisperConfig

        config = Config(
            engine=EngineConfig(name="whisper", language="en"),
            whisper=WhisperConfig(model="tiny", device="cpu"),
        )
        engine = WhisperEngine(config)

        transcript = engine.transcribe(sample_video)

        assert transcript.engine == "whisper"
        assert transcript.language == "en"
        assert len(transcript.segments) >= 0

    def test_generate_srt_from_transcript(self, sample_video: Path, tmp_path: Path) -> None:
        """Test SRT generation from transcript."""
        from vidsub.models import (
            CanonicalTranscript,
            Segment,
            SubtitleConfig,
        )
        from vidsub.subtitles import generate_srt

        transcript = CanonicalTranscript(
            engine="whisper",
            model="tiny",
            language="en",
            duration_sec=5.0,
            segments=[
                Segment(start=0.0, end=2.0, text="Hello world"),
                Segment(start=2.5, end=4.5, text="Second caption"),
            ],
        )
        config = SubtitleConfig()

        srt_content = generate_srt(transcript, config)

        assert "Hello world" in srt_content
        assert "Second caption" in srt_content
        assert "00:00:00,000 --> 00:00:02,000" in srt_content

    def test_burn_subtitles(self, sample_video: Path, tmp_path: Path) -> None:
        """Test subtitle burn-in produces valid video."""
        from vidsub.burn import burn_subtitles
        from vidsub.models import AssStyleConfig, Config, SubtitleConfig

        # Create a simple ASS subtitle file
        ass_content = """[Script Info]
Title: Test
ScriptType: v4.00+
PlayResX: 640
PlayResY: 480

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,20,&H00FFFFFF,&H000000FF,&H00000000,&H00000000,0,0,0,0,100,100,0,0,1,2,0,2,10,10,10,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
Dialogue: 0,0:00:01.00,0:00:03.00,Default,,0,0,0,,Test Subtitle
"""
        subs_path = tmp_path / "test.ass"
        subs_path.write_text(ass_content)

        output_path = tmp_path / "burned.mp4"
        from vidsub.config import DEFAULT_YOUTUBE_PRESET
        from vidsub.models import FFmpegConfig
        config = Config(
            subtitles=SubtitleConfig(),
            style_ass=AssStyleConfig(),
            ffmpeg=FFmpegConfig(
                default_preset="youtube_1080p",
                presets={"youtube_1080p": DEFAULT_YOUTUBE_PRESET},
            ),
        )

        result = burn_subtitles(sample_video, subs_path, output_path, config)

        assert result.exists()

        # Verify output video
        info = probe_video(result)
        assert info.duration_sec > 0


class TestCLITests:
    """CLI integration tests using typer.testing."""

    def test_cli_help(self) -> None:
        """Test CLI help command works."""
        from typer.testing import CliRunner

        from vidsub.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "vidsub" in result.output.lower()

    def test_init_config_command(self, tmp_path: Path) -> None:
        """Test init-config creates valid config file."""
        from typer.testing import CliRunner

        from vidsub.cli import app

        runner = CliRunner()
        config_path = tmp_path / "vidsub.yaml"

        result = runner.invoke(app, ["init-config", "--path", str(config_path)])

        assert result.exit_code == 0
        assert config_path.exists()
        content = config_path.read_text()
        assert "app:" in content
        assert "engine:" in content
