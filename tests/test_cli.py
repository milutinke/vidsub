"""Tests for CLI commands."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from typer.testing import CliRunner

from vidsub.cli import app
from vidsub.models import Config, EngineConfig

runner = CliRunner()


class TestInitConfig:
    """Tests for init-config command."""

    def test_init_config_creates_file(self, tmp_path: Path) -> None:
        config_path = tmp_path / "test.yaml"
        result = runner.invoke(app, ["init-config", "--path", str(config_path)])
        assert result.exit_code == 0
        assert config_path.exists()
        assert "vidsub" in config_path.read_text()

    def test_init_config_prompts_on_overwrite(self, tmp_path: Path) -> None:
        config_path = tmp_path / "test.yaml"
        config_path.write_text("existing")
        result = runner.invoke(app, ["init-config", "--path", str(config_path)], input="n\n")
        assert result.exit_code != 0  # Aborted


class TestRunCommand:
    """Tests for run command."""

    def test_run_without_video_fails(self) -> None:
        result = runner.invoke(app, ["run"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output or "VIDEO" in result.output

    def test_run_forwards_whisper_model_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        video_path = tmp_path / "video.mp4"
        video_path.write_text("fake")
        captured: dict[str, object] = {}

        def fake_load_config(
            config_path: Path | None,
            overrides: dict[str, Any] | None,
        ) -> Config:
            captured["config_path"] = config_path
            captured["overrides"] = overrides
            return Config()

        def fake_run_pipeline(
            video: Path,
            config: Config,
            preset_name: str | None = None,
        ) -> SimpleNamespace:
            del video, config, preset_name
            return SimpleNamespace(
                transcript_path=tmp_path / "transcript.json",
                srt_path=None,
                ass_path=None,
                burned_video_path=None,
            )

        monkeypatch.setattr("vidsub.cli.load_config", fake_load_config)
        monkeypatch.setattr("vidsub.cli.run_pipeline", fake_run_pipeline)

        result = runner.invoke(
            app,
            [
                "run",
                str(video_path),
                "--engine",
                "whisper",
                "--whisper-model",
                "sam8000/whisper-large-v3-turbo-serbian-serbia",
            ],
        )

        assert result.exit_code == 0
        assert captured["overrides"] == {
            "engine": {"name": "whisper"},
            "app": {
                "overwrite": False,
                "keep_temp": False,
                "burn": False,
                "verbose_postprocessing": False,
                "show_progress": True,
            },
            "whisper": {
                "model": "sam8000/whisper-large-v3-turbo-serbian-serbia",
            },
        }

    def test_run_rejects_whisper_model_with_gemini_engine(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        video_path = tmp_path / "video.mp4"
        video_path.write_text("fake")

        def fake_load_config(
            config_path: Path | None,
            overrides: dict[str, Any] | None,
        ) -> Config:
            del config_path, overrides
            return Config(engine=EngineConfig(name="gemini"))

        monkeypatch.setattr(
            "vidsub.cli.load_config",
            fake_load_config,
        )

        result = runner.invoke(
            app,
            [
                "run",
                str(video_path),
                "--engine",
                "gemini",
                "--whisper-model",
                "sam8000/whisper-large-v3-turbo-serbian-serbia",
            ],
        )

        assert result.exit_code != 0
        assert "whisper model" in result.output.lower()
        assert "gemini" in result.output.lower()


class TestValidateCommand:
    """Tests for validate command."""

    def test_validate_unsupported_format(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "test.txt"
        bad_file.write_text("content")
        result = runner.invoke(app, ["validate", str(bad_file)])
        assert result.exit_code != 0
        assert "Unsupported" in result.output

    def test_validate_valid_transcript(self, tmp_path: Path) -> None:
        from vidsub.models import CanonicalTranscript, Segment

        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=10.0,
            segments=[Segment(start=0.0, end=5.0, text="Hello")],
        )
        json_file = tmp_path / "test.json"
        json_file.write_text(transcript.model_dump_json())

        result = runner.invoke(app, ["validate", str(json_file)])
        assert result.exit_code == 0
        assert "Valid transcript" in result.output


class TestTranscribeCommand:
    """Tests for transcribe command."""

    def test_transcribe_forwards_whisper_model_override(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        video_path = tmp_path / "video.mp4"
        video_path.write_text("fake")
        captured: dict[str, object] = {}

        def fake_load_config(
            config_path: Path | None,
            overrides: dict[str, Any] | None,
        ) -> Config:
            captured["config_path"] = config_path
            captured["overrides"] = overrides
            return Config()

        def fake_run_pipeline(
            video: Path,
            config: Config,
            preset_name: str | None = None,
        ) -> SimpleNamespace:
            del video, config, preset_name
            return SimpleNamespace(
                transcript_path=tmp_path / "transcript.json",
                srt_path=None,
                ass_path=None,
                burned_video_path=None,
            )

        monkeypatch.setattr("vidsub.cli.load_config", fake_load_config)
        monkeypatch.setattr("vidsub.cli.run_pipeline", fake_run_pipeline)

        result = runner.invoke(
            app,
            [
                "transcribe",
                str(video_path),
                "--whisper-model",
                "sam8000/whisper-large-v3-turbo-serbian-serbia",
            ],
        )

        assert result.exit_code == 0
        assert captured["overrides"] == {
            "app": {
                "overwrite": False,
                "keep_temp": False,
                "burn": False,
                "verbose_postprocessing": False,
                "show_progress": True,
            },
            "whisper": {
                "model": "sam8000/whisper-large-v3-turbo-serbian-serbia",
            },
        }
