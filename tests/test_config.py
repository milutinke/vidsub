"""Tests for configuration loading."""

from pathlib import Path

import pytest

from vidsub.config import find_config_file, load_config, load_yaml_config
from vidsub.models import Config


class TestFindConfigFile:
    """Tests for config file discovery."""

    def test_explicit_path(self, tmp_path: Path) -> None:
        config_file = tmp_path / "custom.yaml"
        config_file.write_text("app:\n  out_dir: ./test\n")
        result = find_config_file(config_file)
        assert result == config_file

    def test_explicit_path_missing_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_config_file(tmp_path / "missing.yaml")

    def test_default_search_no_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        result = find_config_file()
        assert result is None

    def test_default_search_finds_vidsub_yaml(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        config_file = tmp_path / "vidsub.yaml"
        config_file.write_text("app:\n  out_dir: ./test\n")
        result = find_config_file()
        assert result is not None
        assert result.name == config_file.name


class TestLoadYamlConfig:
    """Tests for YAML config loading."""

    def test_valid_yaml(self, tmp_path: Path) -> None:
        config_file = tmp_path / "test.yaml"
        config_file.write_text("app:\n  out_dir: ./test\n")
        result = load_yaml_config(config_file)
        assert result["app"]["out_dir"] == "./test"

    def test_invalid_yaml_raises(self, tmp_path: Path) -> None:
        config_file = tmp_path / "test.yaml"
        config_file.write_text("invalid: yaml: [")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_yaml_config(config_file)


class TestLoadConfig:
    """Tests for full config loading with precedence."""

    def test_defaults_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("VIDSUB_ENGINE", raising=False)
        config = load_config()
        assert isinstance(config, Config)
        assert config.app.out_dir == "./out"
        assert config.engine.name == "whisper"
        assert config.gemini.chunk_seconds == 60
        assert config.gemini.concurrency == 3

    def test_cli_overrides_take_precedence(self, tmp_path: Path) -> None:
        config_file = tmp_path / "vidsub.yaml"
        config_file.write_text("app:\n  out_dir: ./from_yaml\n")

        overrides = {"app": {"out_dir": "./from_cli"}}
        config = load_config(config_file, overrides)
        assert config.app.out_dir == "./from_cli"
