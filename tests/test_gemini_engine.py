"""Tests for Gemini transcription engine."""

from typing import Any

import pytest

from vidsub.engines.gemini_engine import (
    GeminiEngine,
    GeminiError,
    _build_prompt,
)
from vidsub.models import Config, EngineConfig, GeminiConfig, Segment


class TestBuildPrompt:
    """Tests for prompt building."""

    def test_includes_chunk_times(self) -> None:
        prompt = _build_prompt(30.0, 10.0)
        assert "30.0" in prompt
        assert "40.0" in prompt

    def test_includes_timestamp_requirements(self) -> None:
        prompt = _build_prompt(0.0, 30.0)
        assert "timestamps must be within" in prompt.lower()
        assert "[0.0, 30.0]" in prompt


class TestGeminiEngine:
    """Tests for GeminiEngine."""

    @pytest.fixture
    def config(self) -> Config:
        return Config(
            engine=EngineConfig(name="gemini", language="en"),
            gemini=GeminiConfig(chunk_seconds=30, overlap_seconds=2, max_retries=2),
        )

    @pytest.fixture
    def engine(self, config: Config, monkeypatch: pytest.MonkeyPatch) -> GeminiEngine:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        return GeminiEngine(config)

    def test_name(self, engine: GeminiEngine) -> None:
        assert engine.name == "gemini"

    def test_get_api_key_from_env(
        self, engine: GeminiEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "my-api-key")
        key = engine._get_api_key()
        assert key == "my-api-key"

    def test_get_api_key_missing_raises(
        self, engine: GeminiEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        with pytest.raises(GeminiError, match="API key not found"):
            engine._get_api_key()

    def test_merge_transcripts(self, engine: GeminiEngine) -> None:
        transcripts: list[dict[str, Any]] = [
            {
                "segments": [
                    {"start": 0.0, "end": 5.0, "text": "First"},
                ],
                "language": "en",
            },
            {
                "segments": [
                    {"start": 5.0, "end": 10.0, "text": "Second"},
                ],
            },
        ]

        result = engine._merge_transcripts(transcripts, 10.0)

        assert result.engine == "gemini"
        assert result.language == "en"
        assert len(result.segments) == 2
        assert result.segments[0].text == "First"
        assert result.segments[1].text == "Second"

    def test_merge_sorts_by_start_time(self, engine: GeminiEngine) -> None:
        transcripts: list[dict[str, Any]] = [
            {
                "segments": [
                    {"start": 10.0, "end": 15.0, "text": "Second"},
                ],
            },
            {
                "segments": [
                    {"start": 0.0, "end": 5.0, "text": "First"},
                ],
            },
        ]

        result = engine._merge_transcripts(transcripts, 15.0)

        assert result.segments[0].start == 0.0
        assert result.segments[1].start == 10.0

    def test_remove_overlaps_no_overlap(self, engine: GeminiEngine) -> None:
        segments = [
            Segment(start=0.0, end=5.0, text="First"),
            Segment(start=5.0, end=10.0, text="Second"),
        ]
        result = engine._remove_overlaps(segments)
        assert len(result) == 2

    def test_remove_overlaps_with_overlap(self, engine: GeminiEngine) -> None:
        segments = [
            Segment(start=0.0, end=5.0, text="First"),
            Segment(start=3.0, end=8.0, text="Second"),  # Overlaps
        ]
        result = engine._remove_overlaps(segments)
        assert len(result) == 1
        assert result[0].text == "First"
