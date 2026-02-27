"""Tests for Gemini transcription engine."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from google.genai import errors

from vidsub.engines.gemini_engine import (
    TRANSCRIPT_SCHEMA,
    ChunkJob,
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
        assert "timestamps must stay within" in prompt.lower()
        assert "[0.0, 30.0]" in prompt

    def test_requires_json_only_response(self) -> None:
        prompt = _build_prompt(0.0, 30.0)
        assert "respond only with valid json" in prompt.lower()

    def test_mentions_overlap_deduplication(self) -> None:
        prompt = _build_prompt(58.0, 60.0)
        assert "overlap" in prompt.lower()
        assert "do not repeat speech" in prompt.lower()

    def test_uses_mmss_clip_range(self) -> None:
        prompt = _build_prompt(30.0, 10.0)
        assert "00:30" in prompt
        assert "00:40" in prompt

    def test_mentions_audible_onset_and_offset_alignment(self) -> None:
        prompt = _build_prompt(0.0, 30.0)
        assert "first spoken word becomes audible" in prompt.lower()
        assert "last spoken word finishes" in prompt.lower()

    def test_includes_internal_verification_step(self) -> None:
        prompt = _build_prompt(0.0, 30.0)
        assert "before responding, internally verify" in prompt.lower()

    def test_includes_alignment_example(self) -> None:
        prompt = _build_prompt(0.0, 30.0)
        assert "example alignment" in prompt.lower()
        assert "do not shift it earlier or later" in prompt.lower()


class TestTranscriptSchema:
    """Tests for transcript schema guidance."""

    def test_schema_describes_audible_boundaries(self) -> None:
        schema_properties = cast(dict[str, Any], TRANSCRIPT_SCHEMA["properties"])
        segment_properties = cast(
            dict[str, Any],
            cast(dict[str, Any], cast(dict[str, Any], schema_properties["segments"])["items"])[
                "properties"
            ],
        )
        start_description = cast(dict[str, str], segment_properties["start"])["description"]
        end_description = cast(dict[str, str], segment_properties["end"])["description"]

        assert "first spoken word" in start_description.lower()
        assert "last spoken word" in end_description.lower()


class TestGeminiEngine:
    """Tests for GeminiEngine."""

    @pytest.fixture
    def config(self) -> Config:
        return Config(
            engine=EngineConfig(name="gemini", language="en"),
            gemini=GeminiConfig(
                chunk_seconds=60,
                overlap_seconds=2,
                max_retries=2,
                concurrency=2,
                upload_timeout_sec=2,
                poll_interval_sec=0.01,
                retry_base_delay_sec=0.01,
                retry_max_delay_sec=0.02,
            ),
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

    def test_build_chunk_jobs_drops_tiny_trailing_chunk(
        self, engine: GeminiEngine, tmp_path: Path
    ) -> None:
        jobs = engine._build_chunk_jobs(duration_sec=120.0, temp_dir=tmp_path)

        assert [(job.index, job.start_sec, job.end_sec) for job in jobs] == [
            (0, 0.0, 60.0),
            (1, 58.0, 118.0),
        ]
        assert jobs[0].chunk_path == tmp_path / "chunk_0000.mp4"

    def test_normalize_chunk_clamps_out_of_range_timestamps(self, engine: GeminiEngine) -> None:
        normalized = engine._normalize_chunk_transcript(
            {
                "segments": [
                    {"start": 9.0, "end": 12.0, "text": "  first  "},
                    {"start": 12.0, "end": 25.0, "text": "second"},
                    {"start": -1.0, "end": 5.0, "text": "drop me"},
                    {"start": 14.0, "end": 14.0, "text": "also drop"},
                    {"start": 15.0, "end": 18.0, "text": "   "},
                ],
                "language": "en",
            },
            chunk_start=10.0,
            chunk_end=20.0,
        )

        assert normalized == {
            "language": "en",
            "segments": [
                {"start": 10.0, "end": 12.0, "text": "first"},
                {"start": 12.0, "end": 20.0, "text": "second"},
            ],
        }

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

    def test_merge_drops_exact_overlap_duplicates(self, engine: GeminiEngine) -> None:
        transcripts: list[dict[str, Any]] = [
            {
                "segments": [
                    {"start": 0.0, "end": 5.0, "text": "Hello world"},
                ],
                "language": "en",
            },
            {
                "segments": [
                    {"start": 4.0, "end": 6.0, "text": " Hello world "},
                ],
                "language": "en",
            },
        ]

        result = engine._merge_transcripts(transcripts, 10.0)

        assert len(result.segments) == 1
        assert result.segments[0].text == "Hello world"

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
            Segment(start=3.0, end=8.0, text="Second"),
        ]
        result = engine._remove_overlaps(segments)
        assert len(result) == 1
        assert result[0].text == "First"

    @pytest.mark.asyncio
    async def test_retryable_error_retries_then_succeeds(
        self, engine: GeminiEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = 0

        async def fake_sleep(_: float) -> None:
            return None

        async def operation() -> str:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise errors.ServerError(503, {}, None)
            return "ok"

        monkeypatch.setattr("vidsub.engines.gemini_engine.asyncio.sleep", fake_sleep)

        result = await engine._retry_async(operation, "generate_content")

        assert result == "ok"
        assert attempts == 3

    @pytest.mark.asyncio
    async def test_non_retryable_error_fails_immediately(
        self, engine: GeminiEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = 0

        async def fake_sleep(_: float) -> None:
            return None

        async def operation() -> str:
            nonlocal attempts
            attempts += 1
            raise errors.ClientError(400, {}, None)

        monkeypatch.setattr("vidsub.engines.gemini_engine.asyncio.sleep", fake_sleep)

        with pytest.raises(errors.ClientError):
            await engine._retry_async(operation, "generate_content")

        assert attempts == 1

    @pytest.mark.asyncio
    async def test_invalid_json_retries_until_limit(
        self, engine: GeminiEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        responses = [
            SimpleNamespace(text="{not valid json"),
            SimpleNamespace(text='{"segments": [], "language": "en"}'),
        ]

        async def fake_sleep(_: float) -> None:
            return None

        async def generate_content(**_: Any) -> Any:
            return responses.pop(0)

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=generate_content),
        )

        monkeypatch.setattr("vidsub.engines.gemini_engine.asyncio.sleep", fake_sleep)
        monkeypatch.setattr(engine, "_get_async_client", lambda: fake_client)

        result = await engine._transcribe_uploaded_chunk(
            cast(Any, SimpleNamespace(uri="files/test", mime_type="video/mp4")),
            chunk_start=0.0,
            chunk_duration=10.0,
        )

        assert result == {"segments": [], "language": "en"}

    @pytest.mark.asyncio
    async def test_transcribe_uploaded_chunk_requires_file_uri(
        self, engine: GeminiEngine
    ) -> None:
        with pytest.raises(GeminiError, match="file URI"):
            await engine._transcribe_uploaded_chunk(
                cast(Any, SimpleNamespace(uri=None, mime_type="video/mp4")),
                chunk_start=0.0,
                chunk_duration=10.0,
            )

    @pytest.mark.asyncio
    async def test_transcribe_uploaded_chunk_places_video_before_text(
        self, engine: GeminiEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        captured: dict[str, Any] = {}

        async def fake_generate_content(**kwargs: Any) -> Any:
            captured.update(kwargs)
            return SimpleNamespace(text='{"segments": [], "language": "en"}')

        fake_client = SimpleNamespace(
            models=SimpleNamespace(generate_content=fake_generate_content),
        )

        monkeypatch.setattr(engine, "_get_async_client", lambda: fake_client)

        await engine._transcribe_uploaded_chunk(
            cast(Any, SimpleNamespace(uri="files/test", mime_type="video/mp4")),
            chunk_start=30.0,
            chunk_duration=10.0,
        )

        parts = captured["contents"][0].parts
        assert getattr(parts[0], "file_data", None) is not None
        assert getattr(parts[1], "text", None) is not None

    @pytest.mark.asyncio
    async def test_upload_poll_timeout_raises_gemini_error(
        self, engine: GeminiEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_obj = SimpleNamespace(
            name="files/123",
            uri="files/123",
            state=SimpleNamespace(name="PROCESSING"),
        )
        calls = 0
        clock_values = [0.0, 0.0, 1.5, 2.5, 3.5]
        clock_index = 0

        async def fake_sleep(_: float) -> None:
            return None

        async def get(**_: Any) -> Any:
            nonlocal calls
            calls += 1
            return file_obj

        def fake_monotonic() -> float:
            nonlocal clock_index
            value = clock_values[min(clock_index, len(clock_values) - 1)]
            clock_index += 1
            return value

        fake_client = SimpleNamespace(files=SimpleNamespace(get=get))

        monkeypatch.setattr("vidsub.engines.gemini_engine.asyncio.sleep", fake_sleep)
        monkeypatch.setattr("vidsub.engines.gemini_engine.time.monotonic", fake_monotonic)
        monkeypatch.setattr(engine, "_get_async_client", lambda: fake_client)

        with pytest.raises(GeminiError, match="ACTIVE"):
            await engine._wait_for_file_active(cast(Any, file_obj))

        assert calls >= 1

    @pytest.mark.asyncio
    async def test_failed_file_state_raises_without_extra_retries(
        self, engine: GeminiEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        file_obj = SimpleNamespace(
            name="files/123",
            uri="files/123",
            state=SimpleNamespace(name="FAILED"),
        )
        calls = 0

        async def get(**_: Any) -> Any:
            nonlocal calls
            calls += 1
            return file_obj

        fake_client = SimpleNamespace(files=SimpleNamespace(get=get))
        monkeypatch.setattr(engine, "_get_async_client", lambda: fake_client)

        with pytest.raises(GeminiError, match="FAILED"):
            await engine._wait_for_file_active(cast(Any, file_obj))

        assert calls == 0

    @pytest.mark.asyncio
    async def test_delete_file_attempted_after_generation_failure(
        self, engine: GeminiEngine, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        chunk_path = tmp_path / "chunk_0000.mp4"
        chunk_path.write_text("video")
        deleted: list[str] = []

        job = ChunkJob(
            index=0,
            start_sec=0.0,
            end_sec=10.0,
            duration_sec=10.0,
            chunk_path=chunk_path,
        )

        async def fake_upload(_: Path) -> Any:
            return SimpleNamespace(name="files/abc", uri="files/abc", mime_type="video/mp4")

        async def fake_transcribe(
            _: Any,
            chunk_start: float,
            chunk_duration: float,
        ) -> dict[str, Any]:
            assert chunk_start == 0.0
            assert chunk_duration == 10.0
            raise GeminiError("boom")

        async def fake_delete(file_name: str) -> None:
            deleted.append(file_name)

        async def passthrough_to_thread(func: Any, *args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        monkeypatch.setattr(engine, "_extract_chunk", lambda *args: None)
        monkeypatch.setattr(engine, "_upload_video_async", fake_upload)
        monkeypatch.setattr(engine, "_transcribe_uploaded_chunk", fake_transcribe)
        monkeypatch.setattr(engine, "_delete_file_async", fake_delete)
        monkeypatch.setattr("vidsub.engines.gemini_engine.asyncio.to_thread", passthrough_to_thread)

        with pytest.raises(GeminiError, match="boom"):
            await engine._process_chunk_job(tmp_path / "video.mp4", job)

        assert deleted == ["files/abc"]
        assert not chunk_path.exists()

    @pytest.mark.asyncio
    async def test_chunked_transcription_respects_configured_concurrency(
        self, engine: GeminiEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        jobs = [
            ChunkJob(i, float(i * 10), float(i * 10 + 10), 10.0, Path(f"chunk_{i}.mp4"))
            for i in range(4)
        ]
        active = 0
        max_active = 0

        async def fake_process_chunk_job(_: Path, job: ChunkJob) -> tuple[int, dict[str, Any]]:
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            active -= 1
            return job.index, {"language": "en", "segments": []}

        monkeypatch.setattr(engine, "_process_chunk_job", fake_process_chunk_job)

        results = await engine._run_chunk_jobs(Path("video.mp4"), jobs)

        assert max_active == engine.config.gemini.concurrency
        assert len(results) == len(jobs)

    @pytest.mark.asyncio
    async def test_parallel_results_are_sorted_by_chunk_index(
        self, engine: GeminiEngine, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        jobs = [
            ChunkJob(i, float(i * 10), float(i * 10 + 10), 10.0, Path(f"chunk_{i}.mp4"))
            for i in range(3)
        ]

        async def fake_process_chunk_job(_: Path, job: ChunkJob) -> tuple[int, dict[str, Any]]:
            if job.index == 0:
                await asyncio.sleep(0.02)
            elif job.index == 1:
                await asyncio.sleep(0.01)
            return job.index, {
                "language": "en",
                "segments": [
                    {"start": job.start_sec, "end": job.end_sec, "text": str(job.index)}
                ],
            }

        monkeypatch.setattr(engine, "_process_chunk_job", fake_process_chunk_job)

        results = await engine._run_chunk_jobs(Path("video.mp4"), jobs)

        assert [item["segments"][0]["text"] for item in results] == ["0", "1", "2"]
