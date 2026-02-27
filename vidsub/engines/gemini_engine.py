"""Gemini transcription engine using the Google Gemini API."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import shutil
import subprocess
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar, cast

from vidsub.engines.base import TranscriptionEngine, TranscriptionError
from vidsub.ffmpeg import probe_video
from vidsub.models import CanonicalTranscript, Segment

if TYPE_CHECKING:
    from google.genai.client import AsyncClient, Client
    from google.genai.types import File


logger = logging.getLogger(__name__)


class GeminiError(TranscriptionError):
    """Exception for Gemini-specific errors."""


class _RetryableGeminiError(GeminiError):
    """Internal exception for transient Gemini failures."""


@dataclass(slots=True)
class ChunkJob:
    """Represents a video chunk to process."""

    index: int
    start_sec: float
    end_sec: float
    duration_sec: float
    chunk_path: Path


T = TypeVar("T")


TRANSCRIPT_SCHEMA = {
    "type": "object",
    "properties": {
        "segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {
                        "type": "number",
                        "description": "Start time in seconds from the full video timeline",
                    },
                    "end": {
                        "type": "number",
                        "description": "End time in seconds from the full video timeline",
                    },
                    "text": {
                        "type": "string",
                        "description": "Transcribed text for this segment",
                    },
                },
                "required": ["start", "end", "text"],
            },
        },
        "language": {
            "type": "string",
            "description": "ISO 639-1 language code (for example 'en')",
        },
    },
    "required": ["segments", "language"],
}


def _build_prompt(chunk_start: float, chunk_duration: float) -> str:
    """Build a transcription prompt for Gemini.

    Args:
        chunk_start: Start time of the chunk in the full video timeline.
        chunk_duration: Duration of the chunk in seconds.

    Returns:
        Prompt string for Gemini.
    """
    chunk_end = chunk_start + chunk_duration
    return f"""Transcribe the spoken audio from this video clip.

This clip covers the full-video time range {chunk_start:.1f}s to {chunk_end:.1f}s.

Return one JSON object with this structure:
- language: ISO 639-1 language code
- segments: array of objects with:
  - start: segment start time in seconds relative to the full video
  - end: segment end time in seconds relative to the full video
  - text: verbatim spoken text

Rules:
- Timestamps must stay within [{chunk_start:.1f}, {chunk_end:.1f}]
- Segments must be chronological and non-overlapping
- Use proper punctuation and capitalization
- Prefer coherent phrase-level segments over many tiny fragments
- If this chunk overlaps adjacent chunks, do not repeat speech that appears in the overlap
- Do not include silence, music-only spans, or non-speech filler descriptions

Respond only with valid JSON matching the schema.
"""


class GeminiEngine(TranscriptionEngine):
    """Gemini transcription engine using Gemini Files + structured JSON output."""

    _client: Client | None = None

    @property
    def name(self) -> str:
        """Return the engine identifier."""

        return "gemini"

    def _get_api_key(self) -> str:
        """Get Gemini API key from the configured environment variable.

        Returns:
            Gemini API key.

        Raises:
            GeminiError: If the API key is missing.
        """
        env_var = self.config.gemini.api_key_env
        api_key = os.environ.get(env_var)
        if not api_key:
            raise GeminiError(
                f"Gemini API key not found. Set the {env_var} environment variable."
            )
        return api_key

    def _get_client(self) -> Client:
        """Create or return the cached Gemini client."""

        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._get_api_key())
            logger.debug("Created Gemini client")
        return self._client

    def _get_async_client(self) -> AsyncClient:
        """Return the async Gemini client."""

        return self._get_client().aio

    def transcribe(self, input_path: Path) -> CanonicalTranscript:
        """Transcribe a video with Gemini.

        Args:
            input_path: Path to the input video.

        Returns:
            Canonical transcript for the video.
        """
        self.validate_input(input_path)
        duration_sec = probe_video(input_path).duration_sec

        if duration_sec <= self.config.gemini.chunk_seconds:
            return asyncio.run(self._transcribe_whole_async(input_path, duration_sec))

        return asyncio.run(self._transcribe_chunked_async(input_path, duration_sec))

    async def _transcribe_whole_async(
        self, video_path: Path, duration_sec: float
    ) -> CanonicalTranscript:
        """Transcribe a short video in a single request."""

        uploaded_file: File | None = None
        try:
            uploaded_file = await self._upload_video_async(video_path)
            transcript = await self._transcribe_uploaded_chunk(
                uploaded_file, chunk_start=0.0, chunk_duration=duration_sec
            )
            return self._merge_transcripts([transcript], duration_sec)
        finally:
            if uploaded_file is not None:
                await self._delete_file_async(self._require_file_name(uploaded_file.name))

    async def _transcribe_chunked_async(
        self, video_path: Path, duration_sec: float
    ) -> CanonicalTranscript:
        """Transcribe a long video using parallel chunk processing."""

        temp_dir = Path(tempfile.mkdtemp(prefix="vidsub_gemini_"))
        jobs = self._build_chunk_jobs(duration_sec, temp_dir)
        logger.info("Transcribing in %s chunks", len(jobs))

        try:
            transcripts = await self._run_chunk_jobs(video_path, jobs)
        finally:
            if not self.config.app.keep_temp:
                shutil.rmtree(temp_dir, ignore_errors=True)

        return self._merge_transcripts(transcripts, duration_sec)

    def _build_chunk_jobs(self, duration_sec: float, temp_dir: Path) -> list[ChunkJob]:
        """Build chunk jobs for a long video."""

        chunk_size = float(self.config.gemini.chunk_seconds)
        overlap = float(self.config.gemini.overlap_seconds)
        step = chunk_size - overlap
        chunk_starts: list[float] = []
        start = 0.0

        while start < duration_sec:
            chunk_starts.append(start)
            start += step

        if len(chunk_starts) > 1 and duration_sec - chunk_starts[-1] < 5:
            chunk_starts.pop()

        jobs: list[ChunkJob] = []
        for index, chunk_start in enumerate(chunk_starts):
            chunk_end = min(chunk_start + chunk_size, duration_sec)
            jobs.append(
                ChunkJob(
                    index=index,
                    start_sec=chunk_start,
                    end_sec=chunk_end,
                    duration_sec=chunk_end - chunk_start,
                    chunk_path=temp_dir / f"chunk_{index:04d}.mp4",
                )
            )
        return jobs

    async def _run_chunk_jobs(
        self, video_path: Path, jobs: list[ChunkJob]
    ) -> list[dict[str, Any]]:
        """Run chunk jobs with bounded concurrency."""

        semaphore = asyncio.Semaphore(self.config.gemini.concurrency)

        async def run_job(job: ChunkJob) -> tuple[int, dict[str, Any]]:
            async with semaphore:
                return await self._process_chunk_job(video_path, job)

        tasks = [asyncio.create_task(run_job(job)) for job in jobs]
        try:
            indexed_results = await asyncio.gather(*tasks)
        except Exception:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        indexed_results.sort(key=lambda item: item[0])
        return [result for _, result in indexed_results]

    async def _process_chunk_job(
        self, video_path: Path, job: ChunkJob
    ) -> tuple[int, dict[str, Any]]:
        """Extract, upload, transcribe, and clean up a chunk."""

        uploaded_file: File | None = None
        try:
            logger.info(
                "Processing chunk %s: %.1fs-%.1fs",
                job.index + 1,
                job.start_sec,
                job.end_sec,
            )
            await asyncio.to_thread(
                self._extract_chunk,
                video_path,
                job.chunk_path,
                job.start_sec,
                job.duration_sec,
            )
            uploaded_file = await self._upload_video_async(job.chunk_path)
            transcript = await self._transcribe_uploaded_chunk(
                uploaded_file,
                chunk_start=job.start_sec,
                chunk_duration=job.duration_sec,
            )
            return job.index, transcript
        finally:
            if uploaded_file is not None:
                await self._delete_file_async(self._require_file_name(uploaded_file.name))
            if not self.config.app.keep_temp and job.chunk_path.exists():
                await asyncio.to_thread(job.chunk_path.unlink)

    def _extract_chunk(
        self,
        video_path: Path,
        output_path: Path,
        start_sec: float,
        duration_sec: float,
    ) -> None:
        """Extract a video chunk using FFmpeg.

        Args:
            video_path: Source video path.
            output_path: Destination chunk path.
            start_sec: Chunk start time.
            duration_sec: Chunk duration.

        Raises:
            GeminiError: If FFmpeg fails.
        """
        cmd = [
            "ffmpeg",
            "-y",
            "-ss",
            str(start_sec),
            "-t",
            str(duration_sec),
            "-i",
            str(video_path),
            "-c",
            "copy",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise GeminiError(f"Failed to extract chunk: {e.stderr}") from e

    async def _upload_video_async(self, video_path: Path) -> File:
        """Upload a video file and wait until Gemini marks it ACTIVE."""

        client = self._get_async_client()
        mime_type = self._get_mime_type(video_path)

        async def upload() -> File:
            uploaded = await client.files.upload(
                file=str(video_path),
                config={"mime_type": mime_type},
            )
            if not uploaded.uri or not uploaded.name:
                raise GeminiError("Upload succeeded but no file metadata was returned")
            return uploaded

        uploaded_file = await self._retry_async(upload, "file upload")
        return await self._wait_for_file_active(uploaded_file)

    async def _wait_for_file_active(self, file_obj: File) -> File:
        """Poll Gemini until the uploaded file becomes ACTIVE."""

        state_name = self._get_file_state_name(file_obj)
        if state_name == "ACTIVE":
            return file_obj
        if state_name == "FAILED":
            raise GeminiError(f"Gemini file {file_obj.name} entered FAILED state")

        client = self._get_async_client()
        started_at = time.monotonic()
        timeout_sec = self.config.gemini.upload_timeout_sec

        while time.monotonic() - started_at < timeout_sec:
            await asyncio.sleep(self.config.gemini.poll_interval_sec)
            file_name = self._require_file_name(file_obj.name)
            file_obj = await self._retry_async(
                partial(client.files.get, name=file_name),
                "file status poll",
            )
            state_name = self._get_file_state_name(file_obj)
            if state_name == "ACTIVE":
                return file_obj
            if state_name == "FAILED":
                raise GeminiError(f"Gemini file {file_obj.name} entered FAILED state")

        raise GeminiError(f"Gemini file {file_obj.name} did not become ACTIVE in time")

    async def _delete_file_async(self, file_name: str) -> None:
        """Delete an uploaded Gemini file.

        Args:
            file_name: Gemini file name identifier.
        """
        client = self._get_async_client()

        try:
            await self._retry_async(
                lambda: client.files.delete(name=file_name),
                "file delete",
            )
        except Exception as exc:
            logger.warning("Failed to delete Gemini file %s: %s", file_name, exc)

    async def _transcribe_uploaded_chunk(
        self,
        file_obj: File,
        chunk_start: float,
        chunk_duration: float,
    ) -> dict[str, Any]:
        """Transcribe an uploaded file chunk."""

        from google.genai import types

        client = self._get_async_client()
        prompt = _build_prompt(chunk_start, chunk_duration)
        mime_type = getattr(file_obj, "mime_type", None) or "video/mp4"
        file_uri = self._require_file_uri(file_obj.uri)
        generation_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TRANSCRIPT_SCHEMA,
        )

        async def generate_content() -> dict[str, Any]:
            response = await client.models.generate_content(
                model=self.config.gemini.model,
                contents=[
                    types.Content(
                        role="user",
                        parts=[
                            types.Part.from_text(text=prompt),
                            types.Part.from_uri(file_uri=file_uri, mime_type=mime_type),
                        ],
                    )
                ],
                config=generation_config,
            )
            response_text = response.text or ""
            parsed = json.loads(response_text)
            if not isinstance(parsed, dict):
                raise _RetryableGeminiError("Gemini returned JSON that was not an object")
            return cast(dict[str, Any], parsed)

        transcript = await self._retry_async(generate_content, "generate_content")
        return self._normalize_chunk_transcript(
            transcript,
            chunk_start=chunk_start,
            chunk_end=chunk_start + chunk_duration,
        )

    async def _retry_async(
        self,
        operation: Callable[[], Awaitable[T]],
        operation_name: str,
    ) -> T:
        """Retry a transient async operation with exponential backoff."""

        max_retries = self.config.gemini.max_retries
        for attempt in range(max_retries + 1):
            try:
                return await operation()
            except Exception as exc:
                if attempt >= max_retries or not self._is_retryable_error(exc):
                    raise
                delay = self._get_retry_delay(exc, attempt)
                logger.warning(
                    "%s failed (%s). Retrying in %.2fs (%s/%s)",
                    operation_name,
                    exc,
                    delay,
                    attempt + 1,
                    max_retries,
                )
                await asyncio.sleep(delay)

        raise GeminiError(f"{operation_name} exceeded retry budget")

    def _is_retryable_error(self, exc: Exception) -> bool:
        """Return whether an exception should be retried."""

        from google.genai import errors

        if isinstance(exc, (json.JSONDecodeError, _RetryableGeminiError)):
            return True
        if isinstance(exc, errors.ServerError):
            return True
        status_code = self._get_status_code(exc)
        if isinstance(exc, errors.ClientError):
            return status_code == 429
        if isinstance(exc, errors.APIError):
            return status_code in {429, 500, 502, 503, 504}
        return isinstance(exc, (ConnectionError, OSError, TimeoutError))

    def _get_retry_delay(self, exc: Exception, attempt: int) -> float:
        """Compute retry delay with cap, jitter, and optional Retry-After."""

        retry_after = self._extract_retry_after(exc)
        if retry_after is not None:
            return min(retry_after, self.config.gemini.retry_max_delay_sec)

        base_delay = self.config.gemini.retry_base_delay_sec * (2 ** attempt)
        capped_delay = min(base_delay, self.config.gemini.retry_max_delay_sec)
        jitter = capped_delay * 0.2 * random.random()
        max_delay = float(self.config.gemini.retry_max_delay_sec)
        return float(min(capped_delay + jitter, max_delay))

    def _extract_retry_after(self, exc: Exception) -> float | None:
        """Extract a Retry-After delay from an API exception if present."""

        response = getattr(exc, "response", None)
        headers = getattr(response, "headers", None)
        if headers is None:
            return None
        retry_after = headers.get("Retry-After")
        if retry_after is None:
            return None
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            return None

    def _get_status_code(self, exc: Exception) -> int | None:
        """Extract an integer HTTP-like status code from an SDK exception."""

        status = getattr(exc, "status", None)
        if isinstance(status, int):
            return status
        if isinstance(status, str):
            try:
                return int(status)
            except ValueError:
                return None
        return None

    def _normalize_chunk_transcript(
        self,
        transcript: dict[str, Any],
        chunk_start: float,
        chunk_end: float,
    ) -> dict[str, Any]:
        """Clamp and clean a chunk transcript so it is safe to merge."""

        normalized_segments: list[dict[str, Any]] = []
        for raw_segment in transcript.get("segments", []):
            text = str(raw_segment.get("text", "")).strip()
            if not text:
                continue

            start = max(float(raw_segment["start"]), chunk_start)
            end = min(float(raw_segment["end"]), chunk_end)
            if end <= start:
                continue

            normalized_segments.append(
                {
                    "start": start,
                    "end": end,
                    "text": text,
                }
            )

        normalized_segments.sort(key=lambda segment: segment["start"])
        return {
            "language": transcript.get("language", self.config.engine.language),
            "segments": normalized_segments,
        }

    def _merge_transcripts(
        self, transcripts: list[dict[str, Any]], duration_sec: float
    ) -> CanonicalTranscript:
        """Merge chunk transcripts into the canonical transcript."""

        all_segments: list[Segment] = []
        language: str | None = None

        for transcript in transcripts:
            if language is None:
                language = transcript.get("language", self.config.engine.language)

            for segment_data in transcript.get("segments", []):
                all_segments.append(
                    Segment(
                        start=float(segment_data["start"]),
                        end=float(segment_data["end"]),
                        text=str(segment_data["text"]).strip(),
                        speaker=None,
                        words=None,
                    )
                )

        all_segments.sort(key=lambda segment: segment.start)
        merged_segments = self._remove_overlaps(all_segments)

        return CanonicalTranscript(
            engine="gemini",
            model=self.config.gemini.model,
            language=language or self.config.engine.language,
            duration_sec=duration_sec,
            segments=merged_segments,
        )

    def _remove_overlaps(self, segments: list[Segment]) -> list[Segment]:
        """Remove overlapping segments, keeping the earliest segment."""

        if not segments:
            return []

        result = [segments[0]]
        for segment in segments[1:]:
            previous = result[-1]
            if segment.start < previous.end:
                if self._segments_look_duplicate(previous, segment):
                    logger.debug(
                        "Dropping duplicate overlap segment %.2f-%.2f",
                        segment.start,
                        segment.end,
                    )
                else:
                    logger.warning(
                        "Skipping overlapping segment %.2f-%.2f because it overlaps with %.2f-%.2f",
                        segment.start,
                        segment.end,
                        previous.start,
                        previous.end,
                    )
                continue

            result.append(segment)
        return result

    def _segments_look_duplicate(self, left: Segment, right: Segment) -> bool:
        """Return whether two segments likely describe the same overlapped speech."""

        left_text = " ".join(left.text.lower().split())
        right_text = " ".join(right.text.lower().split())
        if left_text == right_text:
            return True
        return left_text.endswith(right_text) or right_text.endswith(left_text)

    def _get_file_state_name(self, file_obj: File) -> str:
        """Return the SDK file state name."""

        state = getattr(file_obj, "state", None)
        if state is None:
            return "STATE_UNSPECIFIED"
        return getattr(state, "name", str(state))

    def _get_mime_type(self, video_path: Path) -> str:
        """Return the MIME type for a video file path."""

        mime_types = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
            ".avi": "video/x-msvideo",
        }
        return mime_types.get(video_path.suffix.lower(), "video/mp4")

    def _require_file_name(self, file_name: str | None) -> str:
        """Return a non-empty Gemini file name."""

        if not file_name:
            raise GeminiError("Gemini file metadata is missing a file name")
        return file_name

    def _require_file_uri(self, file_uri: str | None) -> str:
        """Return a non-empty Gemini file URI."""

        if not file_uri:
            raise GeminiError("Gemini file metadata is missing a file URI")
        return file_uri
