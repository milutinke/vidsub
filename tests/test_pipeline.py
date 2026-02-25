"""Tests for pipeline orchestration."""

from pathlib import Path
from unittest import mock

import pytest

from vidsub.models import (
    AppConfig,
    AssStyleConfig,
    CanonicalTranscript,
    Config,
    EngineConfig,
    Segment,
    SubtitleConfig,
)
from vidsub.pipeline import PipelineResult, run_pipeline


class TestPipelineResult:
    """Tests for PipelineResult."""

    def test_to_dict(self) -> None:
        transcript = CanonicalTranscript(
            engine="whisper",
            model="base",
            language="en",
            duration_sec=10.0,
            segments=[Segment(start=0.0, end=5.0, text="Hello")],
        )
        result = PipelineResult(
            transcript=transcript,
            transcript_path=Path("/out/transcript.json"),
            srt_path=Path("/out/subs.srt"),
        )

        d = result.to_dict()
        assert d["transcript_path"] == "/out/transcript.json"
        assert d["srt_path"] == "/out/subs.srt"
        assert d["ass_path"] is None


class TestRunPipeline:
    """Tests for run_pipeline function."""

    @pytest.fixture
    def config(self, tmp_path: Path) -> Config:
        return Config(
            engine=EngineConfig(name="whisper", language="en"),
            app=AppConfig(
                out_dir=str(tmp_path),
                overwrite=True,
                keep_temp=False,
                burn=False,
            ),
            subtitles=SubtitleConfig(
                formats=["srt"],
                max_chars_per_line=42,
                max_lines=2,
                max_caption_seconds=6.0,
                split_on_silence_ms=350,
                background_style="none",
            ),
            style_ass=AssStyleConfig(
                font_name="Inter",
                font_size=44,
                outline=3,
                shadow=0,
                margin_v=40,
                bg_color="black",
            ),
        )

    @mock.patch("vidsub.pipeline._transcribe")
    def test_pipeline_creates_outputs(
        self,
        mock_transcribe: mock.MagicMock,
        tmp_path: Path,
        config: Config,
    ) -> None:
        # Setup mock transcript
        mock_transcribe.return_value = CanonicalTranscript(
            engine="whisper",
            model="base",
            language="en",
            duration_sec=10.0,
            segments=[Segment(start=0.0, end=5.0, text="Hello world")],
        )

        video = tmp_path / "test.mp4"
        video.write_text("fake")

        # Execute
        result = run_pipeline(video, config)

        # Verify
        assert result.transcript_path is not None
        assert result.srt_path is not None
        assert result.transcript_path.exists()
        assert result.srt_path.exists()

    def test_pipeline_with_existing_transcript(
        self, tmp_path: Path, config: Config
    ) -> None:
        # Create existing transcript
        transcript = CanonicalTranscript(
            engine="whisper",
            model="base",
            language="en",
            duration_sec=10.0,
            segments=[Segment(start=0.0, end=5.0, text="Hello")],
        )
        transcript_path = tmp_path / "existing.json"
        transcript_path.write_text(transcript.model_dump_json())

        video = tmp_path / "test.mp4"
        video.write_text("fake")

        # Execute with skip_transcribe
        result = run_pipeline(
            video, config, skip_transcribe=True, existing_transcript=transcript_path
        )

        # Verify transcript loaded
        assert result.transcript.segments[0].text == "Hello"
