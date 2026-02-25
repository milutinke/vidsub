"""Tests for Whisper transcription engine."""

from pathlib import Path
from unittest import mock

import pytest

from vidsub.engines.base import EngineFactory
from vidsub.engines.whisper_engine import WhisperEngine, _get_device
from vidsub.models import Config, EngineConfig, WhisperConfig


class TestGetDevice:
    """Tests for device selection."""

    def test_explicit_cpu(self) -> None:
        assert _get_device("cpu") == "cpu"

    def test_explicit_cuda(self) -> None:
        assert _get_device("cuda") == "cuda"

    def test_auto_returns_valid_device(self) -> None:
        device = _get_device("auto")
        assert device in ("cpu", "cuda", "mps")


class TestWhisperEngine:
    """Tests for WhisperEngine."""

    @pytest.fixture
    def config(self) -> Config:
        return Config(
            engine=EngineConfig(name="whisper", language="en"),
            whisper=WhisperConfig(model="base", device="cpu"),
        )

    @pytest.fixture
    def engine(self, config: Config) -> WhisperEngine:
        return WhisperEngine(config)

    def test_name(self, engine: WhisperEngine) -> None:
        assert engine.name == "whisper"

    def test_validate_input_missing_file(self, engine: WhisperEngine) -> None:
        with pytest.raises(FileNotFoundError):
            engine.validate_input(Path("/nonexistent.mp4"))

    def test_validate_input_exists(self, engine: WhisperEngine, tmp_path: Path) -> None:
        test_file = tmp_path / "test.mp4"
        test_file.write_text("fake")
        # Should not raise
        engine.validate_input(test_file)

    @mock.patch("vidsub.engines.whisper_engine.extract_audio")
    @mock.patch("vidsub.engines.whisper_engine.probe_video")
    def test_prepare_audio_video_file(
        self,
        mock_probe: mock.MagicMock,
        mock_extract: mock.MagicMock,
        engine: WhisperEngine,
        tmp_path: Path,
    ) -> None:
        # Setup
        video_path = tmp_path / "test.mp4"
        video_path.write_text("fake")
        mock_probe.return_value.duration_sec = 10.0
        mock_extract.return_value = tmp_path / "extracted.wav"

        # Execute
        result = engine._prepare_audio(video_path)

        # Verify
        mock_extract.assert_called_once()
        assert "extract" in str(result).lower() or "audio" in str(result).lower()

    def test_prepare_audio_direct_audio(self, engine: WhisperEngine, tmp_path: Path) -> None:
        audio_path = tmp_path / "test.wav"
        audio_path.write_text("fake")

        result = engine._prepare_audio(audio_path)

        # Should use directly without extraction
        assert result == audio_path

    def test_build_transcribe_options_accurate(self, config: Config) -> None:
        config.whisper.accurate = True
        engine = WhisperEngine(config)
        options = engine._build_transcribe_options()

        assert options["beam_size"] == 5
        assert options["best_of"] == 5

    def test_build_transcribe_options_fast(self, config: Config) -> None:
        config.whisper.accurate = False
        engine = WhisperEngine(config)
        options = engine._build_transcribe_options()

        assert options["beam_size"] == 1
        assert options["best_of"] == 1

    def test_convert_to_canonical(self, engine: WhisperEngine) -> None:
        whisper_result = {
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "Hello world",
                    "words": [
                        {"start": 0.0, "end": 1.0, "text": " Hello", "confidence": 0.95},
                        {"start": 1.0, "end": 2.0, "text": " world", "confidence": 0.92},
                    ],
                }
            ],
            "language": "en",
        }

        result = engine._convert_to_canonical(whisper_result, 10.0)

        assert result.engine == "whisper"
        assert result.language == "en"
        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello world"
        assert result.segments[0].words is not None
        assert len(result.segments[0].words) == 2


class TestEngineFactory:
    """Tests for EngineFactory."""

    def test_create_whisper(self) -> None:
        config = Config(engine=EngineConfig(name="whisper"))
        engine = EngineFactory.create(config)
        assert engine.name == "whisper"

    def test_create_gemini(self) -> None:
        config = Config(engine=EngineConfig(name="gemini"))
        engine = EngineFactory.create(config)
        assert engine.name == "gemini"

    def test_create_unknown_raises(self) -> None:
        # Use a mock object that simulates an invalid config
        mock_config = mock.MagicMock()
        mock_config.engine.name = "unknown"
        with pytest.raises(ValueError, match="Unknown engine"):
            EngineFactory.create(mock_config)
