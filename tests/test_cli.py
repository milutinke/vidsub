"""Tests for CLI commands."""

from pathlib import Path

from typer.testing import CliRunner

from vidsub.cli import app

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
