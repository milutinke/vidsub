"""Gemini transcription engine using Google Gemini 2.5 Pro."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vidsub.engines.base import TranscriptionEngine, TranscriptionError
from vidsub.ffmpeg import probe_video
from vidsub.models import CanonicalTranscript, Segment

if TYPE_CHECKING:
    from google.genai import Client


logger = logging.getLogger(__name__)


class GeminiError(TranscriptionError):
    """Exception for Gemini-specific errors."""
    pass


# JSON Schema for structured transcript output
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
                        "description": "Start time in seconds from video beginning",
                    },
                    "end": {
                        "type": "number",
                        "description": "End time in seconds from video beginning",
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
            "description": "ISO 639-1 language code (e.g., 'en', 'es')",
        },
    },
    "required": ["segments", "language"],
}


def _build_prompt(chunk_start: float, chunk_duration: float) -> str:
    """Build transcription prompt for Gemini.

    Args:
        chunk_start: Start time of this chunk in seconds.
        chunk_duration: Duration of this chunk in seconds.

    Returns:
        Prompt string.
    """
    chunk_end = chunk_start + chunk_duration

    return f"""Transcribe the audio from this video clip.

This clip represents a segment from {chunk_start:.1f}s to {chunk_end:.1f}s of the full video.

Provide the transcript as a JSON object with the following structure:
- segments: array of objects, each with:
  - start: start time in seconds (relative to video beginning, NOT clip start)
  - end: end time in seconds (relative to video beginning, NOT clip start)
  - text: the transcribed text

Important:
- Timestamps must be within the range [{chunk_start:.1f}, {chunk_end:.1f}]
- Segments must be in chronological order
- Text should be verbatim transcription
- Use proper punctuation and capitalization

Respond ONLY with valid JSON matching the schema.
"""


class GeminiEngine(TranscriptionEngine):
    """Gemini transcription engine using Gemini 2.5 Pro.

    Uses the Gemini Files API for video upload and structured JSON output
    for reliable transcript extraction.
    """

    _client: Client | None = None

    @property
    def name(self) -> str:
        return "gemini"

    def _get_api_key(self) -> str:
        """Get Gemini API key from environment.

        Returns:
            API key string.

        Raises:
            GeminiError: If API key not found.
        """
        env_var = self.config.gemini.api_key_env
        api_key = os.environ.get(env_var)

        if not api_key:
            raise GeminiError(
                f"Gemini API key not found. Set the {env_var} environment variable."
            )

        return api_key

    def _get_client(self) -> Client:
        """Get or create Gemini client.

        Returns:
            Configured Gemini client.
        """
        if self._client is None:
            from google.genai import Client

            api_key = self._get_api_key()
            self._client = Client(api_key=api_key)
            logger.debug("Created Gemini client")

        return self._client

    def _upload_video(self, video_path: Path) -> str:
        """Upload video to Gemini Files API.

        Args:
            video_path: Path to video file.

        Returns:
            File URI for uploaded video.

        Raises:
            GeminiError: If upload fails or URI is missing.
        """
        client = self._get_client()

        logger.info(f"Uploading video to Gemini: {video_path}")

        # Upload with mime type detection
        suffix = video_path.suffix.lower()
        mime_types = {
            ".mp4": "video/mp4",
            ".mov": "video/quicktime",
            ".mkv": "video/x-matroska",
            ".webm": "video/webm",
            ".avi": "video/x-msvideo",
        }
        mime_type = mime_types.get(suffix, "video/mp4")

        try:
            file = client.files.upload(
                file=str(video_path),
                config={"mime_type": mime_type},
            )
            uri = file.uri
            if not uri:
                raise GeminiError("Upload succeeded but no URI returned")

            # Wait for file to become ACTIVE
            logger.info(f"Waiting for file to become ACTIVE: {uri}")
            import time

            max_wait = 60  # Maximum 60 seconds wait
            wait_interval = 2  # Check every 2 seconds
            waited = 0

            while waited < max_wait:
                # Check current file state
                if file.state.name == "ACTIVE":
                    logger.info(f"File is ACTIVE after {waited}s")
                    break
                logger.debug(f"File state: {file.state.name}, waiting...")
                time.sleep(wait_interval)
                waited += wait_interval
                # Re-fetch file state
                file = client.files.get(name=file.name)
            else:
                raise GeminiError(
                    f"File did not become ACTIVE within {max_wait}s"
                )

            logger.info(f"Upload complete: {uri}")
            return uri
        except Exception as e:
            raise GeminiError(f"Failed to upload video: {e}") from e

    def _delete_file(self, file_uri: str) -> None:
        """Delete uploaded file from Gemini.

        Args:
            file_uri: URI of file to delete.
        """
        client = self._get_client()

        try:
            # Extract file name from URI
            # URI format: files/FILE_NAME
            file_name = file_uri.split("/")[-1]
            client.files.delete(name=file_name)
            logger.debug(f"Deleted Gemini file: {file_name}")
        except Exception as e:
            logger.warning(f"Failed to delete Gemini file {file_uri}: {e}")

    def transcribe(self, input_path: Path) -> CanonicalTranscript:
        """Transcribe video using Gemini with chunking.

        Args:
            input_path: Path to video file.

        Returns:
            CanonicalTranscript.
        """
        self.validate_input(input_path)

        # Probe video for duration
        video_info = probe_video(input_path)
        duration_sec = video_info.duration_sec

        # Decide on chunking strategy
        chunk_size = self.config.gemini.chunk_seconds
        overlap = self.config.gemini.overlap_seconds

        if duration_sec <= chunk_size:
            # Short video - transcribe whole thing
            return self._transcribe_whole(input_path, duration_sec)

        # Long video - use chunking
        return self._transcribe_chunked(input_path, duration_sec, chunk_size, overlap)

    def _transcribe_whole(
        self, video_path: Path, duration_sec: float
    ) -> CanonicalTranscript:
        """Transcribe entire video at once.

        Args:
            video_path: Path to video file.
            duration_sec: Video duration.

        Returns:
            CanonicalTranscript.
        """
        file_uri = self._upload_video(video_path)

        try:
            transcript = self._transcribe_chunk(file_uri, 0, duration_sec)
            return self._merge_transcripts([transcript], duration_sec)
        finally:
            self._delete_file(file_uri)

    def _transcribe_chunked(
        self,
        video_path: Path,
        duration_sec: float,
        chunk_size: int,
        overlap: int,
    ) -> CanonicalTranscript:
        """Transcribe video in chunks.

        Args:
            video_path: Path to video file.
            duration_sec: Total video duration.
            chunk_size: Chunk size in seconds.
            overlap: Overlap between chunks in seconds.

        Returns:
            Merged CanonicalTranscript.
        """
        import tempfile

        transcripts = []

        # Calculate chunk boundaries
        step = chunk_size - overlap
        chunk_starts = list(range(0, int(duration_sec), step))

        # Remove last chunk if it would be too small
        if len(chunk_starts) > 1 and duration_sec - chunk_starts[-1] < 5:
            chunk_starts.pop()

        logger.info(f"Transcribing in {len(chunk_starts)} chunks")

        temp_dir = Path(tempfile.mkdtemp(prefix="vidsub_gemini_"))

        try:
            for i, start in enumerate(chunk_starts):
                end = min(start + chunk_size, duration_sec)
                actual_chunk_size = end - start

                logger.info(f"Processing chunk {i+1}/{len(chunk_starts)}: {start}-{end}s")

                # Extract chunk
                chunk_path = temp_dir / f"chunk_{i:04d}.mp4"
                self._extract_chunk(video_path, chunk_path, start, actual_chunk_size)

                # Upload and transcribe
                file_uri = self._upload_video(chunk_path)
                try:
                    transcript = self._transcribe_chunk(
                        file_uri, start, actual_chunk_size
                    )
                    transcripts.append(transcript)
                finally:
                    self._delete_file(file_uri)

                # Clean up chunk file
                if not self.config.app.keep_temp:
                    chunk_path.unlink()

        finally:
            # Cleanup temp directory
            if not self.config.app.keep_temp:
                import shutil

                shutil.rmtree(temp_dir, ignore_errors=True)

        return self._merge_transcripts(transcripts, duration_sec)

    def _extract_chunk(
        self,
        video_path: Path,
        output_path: Path,
        start_sec: float,
        duration_sec: float,
    ) -> None:
        """Extract a video chunk using FFmpeg.

        Args:
            video_path: Source video.
            output_path: Output chunk path.
            start_sec: Start time.
            duration_sec: Duration.
        """
        import subprocess

        cmd = [
            "ffmpeg",
            "-y",
            "-ss", str(start_sec),
            "-t", str(duration_sec),
            "-i", str(video_path),
            "-c", "copy",  # Copy streams without re-encoding
            "-avoid_negative_ts", "make_zero",
            str(output_path),
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            raise GeminiError(f"Failed to extract chunk: {e.stderr}") from e

    def _transcribe_chunk(
        self, file_uri: str, chunk_start: float, chunk_duration: float
    ) -> dict[str, Any]:
        """Transcribe a single chunk.

        Args:
            file_uri: URI of uploaded file.
            chunk_start: Start time offset for this chunk.
            chunk_duration: Duration of this chunk.

        Returns:
            Raw transcript dict from Gemini.
        """
        from google.genai import types

        client = self._get_client()
        model = self.config.gemini.model
        max_retries = self.config.gemini.max_retries

        prompt = _build_prompt(chunk_start, chunk_duration)

        generation_config = types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=TRANSCRIPT_SCHEMA,
        )

        for attempt in range(max_retries + 1):
            try:
                response = client.models.generate_content(
                    model=model,
                    contents=[
                        types.Content(
                            role="user",
                            parts=[
                                types.Part.from_text(text=prompt),
                                types.Part.from_uri(file_uri=file_uri, mime_type="video/mp4"),
                            ],
                        )
                    ],
                    config=generation_config,
                )

                # Parse JSON response
                response_text = response.text
                if not response_text:
                    raise GeminiError("Empty response from Gemini")
                result: dict[str, Any] = json.loads(response_text)
                return result

            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    logger.warning(f"Invalid JSON, retrying... ({attempt + 1}/{max_retries})")
                    continue
                raise GeminiError(f"Failed to parse Gemini response as JSON: {e}") from e
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(f"Transcription failed, retrying... ({attempt + 1}/{max_retries})")
                    continue
                raise GeminiError(f"Gemini transcription failed: {e}") from e

        raise GeminiError("Max retries exceeded")

    def _merge_transcripts(
        self, transcripts: list[dict[str, Any]], duration_sec: float
    ) -> CanonicalTranscript:
        """Merge multiple chunk transcripts into one.

        Args:
            transcripts: List of transcript dicts from chunks.
            duration_sec: Total video duration.

        Returns:
            Merged CanonicalTranscript.
        """
        all_segments = []
        language = None

        for transcript in transcripts:
            segments = transcript.get("segments", [])

            # Detect language from first transcript with it
            if language is None:
                language = transcript.get("language", "en")

            for seg in segments:
                segment = Segment(
                    start=float(seg["start"]),
                    end=float(seg["end"]),
                    text=seg["text"].strip(),
                    speaker=None,  # Gemini doesn't do diarization in v1
                    words=None,  # No word-level timestamps from Gemini
                )
                all_segments.append(segment)

        # Sort by start time
        all_segments.sort(key=lambda s: s.start)

        # Remove overlaps (prefer earlier segment)
        merged_segments = self._remove_overlaps(all_segments)

        # Use detected language or config default
        if language is None:
            language = self.config.engine.language

        return CanonicalTranscript(
            engine="gemini",
            model=self.config.gemini.model,
            language=language,
            duration_sec=duration_sec,
            segments=merged_segments,
        )

    def _remove_overlaps(self, segments: list[Segment]) -> list[Segment]:
        """Remove overlapping segments, keeping earlier ones.

        Args:
            segments: Sorted list of segments.

        Returns:
            List with overlaps removed.
        """
        if not segments:
            return []

        result = [segments[0]]

        for seg in segments[1:]:
            prev = result[-1]

            # Check for overlap
            if seg.start < prev.end:
                # Overlap detected - skip this segment
                overlap_duration = prev.end - seg.start
                logger.warning(
                    f"Skipping overlapping segment: {seg.start:.2f}-{seg.end:.2f} "
                    f"(overlaps {overlap_duration:.2f}s with previous)"
                )
                continue

            result.append(seg)

        return result
