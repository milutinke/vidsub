"""Whisper transcription engine using whisper-timestamped."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vidsub.engines.base import TranscriptionEngine, TranscriptionError
from vidsub.ffmpeg import extract_audio, probe_video
from vidsub.models import CanonicalTranscript, Segment, Word
from vidsub.ssl_utils import patched_urlopen

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Whisper models in order of size/accuracy
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]


def _get_device(preferred: str) -> str:
    """Determine compute device based on availability.

    Args:
        preferred: Preferred device ("auto", "cpu", "cuda", "mps").

    Returns:
        Device string to use.
    """
    if preferred != "auto":
        return preferred

    try:
        import torch

        if torch.cuda.is_available():
            logger.info("Using CUDA device")
            return "cuda"
        elif torch.backends.mps.is_available():
            logger.info("Using MPS device (Apple Silicon)")
            return "mps"
    except ImportError:
        pass

    logger.info("Using CPU device")
    return "cpu"


class WhisperEngine(TranscriptionEngine):
    """Whisper transcription engine using whisper-timestamped."""

    _model_cache: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "whisper"

    def _load_model(self) -> Any:
        """Load Whisper model (with caching)."""
        import whisper_timestamped  # type: ignore[import-untyped]

        model_name = self.config.whisper.model

        if model_name not in self._model_cache:
            logger.info(f"Loading Whisper model: {model_name}")
            device = _get_device(self.config.whisper.device)

            # Apply SSL patch for certificate verification on macOS
            with patched_urlopen():
                model = whisper_timestamped.load_model(model_name, device=device)
            self._model_cache[model_name] = model

        return self._model_cache[model_name]

    def transcribe(self, input_path: Path) -> CanonicalTranscript:
        """Transcribe audio/video using whisper-timestamped.

        Args:
            input_path: Path to audio or video file.

        Returns:
            CanonicalTranscript with segments and word-level timestamps.
        """
        self.validate_input(input_path)

        # Probe video for duration
        try:
            video_info = probe_video(input_path)
            duration_sec = video_info.duration_sec
        except Exception as e:
            logger.warning(f"Could not probe video: {e}")
            duration_sec = 0.0

        # Extract audio if needed
        audio_path = self._prepare_audio(input_path)

        try:
            result = self._transcribe_audio(audio_path, duration_sec)
        finally:
            # Cleanup temporary audio if extracted
            if audio_path != input_path and not self.config.app.keep_temp:
                self._cleanup_audio(audio_path)

        return result

    def _prepare_audio(self, input_path: Path) -> Path:
        """Prepare audio file for transcription.

        If input is video, extract audio to temporary WAV file.
        If input is audio, use directly.

        Args:
            input_path: Input file path.

        Returns:
            Path to audio file suitable for Whisper.
        """
        suffix = input_path.suffix.lower()

        # Audio formats that can be used directly
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus"}

        if suffix in audio_extensions:
            logger.debug(f"Using audio file directly: {input_path}")
            return input_path

        # Extract audio from video

        temp_dir = Path(self.config.app.temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)

        audio_path = temp_dir / f"{input_path.stem}_audio.wav"
        logger.info(f"Extracting audio to: {audio_path}")

        return extract_audio(
            input_path,
            audio_path,
            sample_rate=16000,  # Whisper expects 16kHz
            channels=1,  # Mono
        )

    def _cleanup_audio(self, audio_path: Path) -> None:
        """Remove temporary audio file."""
        try:
            audio_path.unlink()
            logger.debug(f"Cleaned up temporary audio: {audio_path}")
        except OSError as e:
            logger.warning(f"Failed to cleanup audio: {e}")

    def _transcribe_audio(
        self,
        audio_path: Path,
        duration_sec: float,
    ) -> CanonicalTranscript:
        """Run whisper-timestamped transcription with progress logging."""
        import whisper_timestamped

        logger.info("[1/4] Loading Whisper model...")
        model = self._load_model()

        logger.info("[2/4] Loading audio...")
        logger.info(f"    Audio file: {audio_path}")
        if duration_sec > 0:
            logger.info(f"    Duration: {duration_sec:.1f}s")

        options = self._build_transcribe_options()

        logger.info("[3/4] Running transcription (this may take a while)...")
        try:
            # Apply SSL patch for VAD model download (torch.hub.load)
            with patched_urlopen():
                result = whisper_timestamped.transcribe(
                    model,
                    str(audio_path),
                    **options,
                )
        except Exception as e:
            raise TranscriptionError(f"Whisper transcription failed: {e}") from e

        logger.info("[4/4] Transcription complete!")
        segments_count = len(result.get("segments", []))
        logger.info(f"    Generated {segments_count} segments")

        return self._convert_to_canonical(result, duration_sec)

    def _build_transcribe_options(self) -> dict[str, Any]:
        """Build options dict for whisper-timestamped.

        Returns:
            Dictionary of transcription options.
        """
        whisper_config = self.config.whisper
        options = {
            "language": self.config.engine.language,
            "verbose": False,
        }

        # VAD option
        if whisper_config.vad:
            options["vad"] = True

        # Accurate mode options
        if whisper_config.accurate:
            options["beam_size"] = 5
            options["best_of"] = 5
            options["temperature"] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
        else:
            options["beam_size"] = 1
            options["best_of"] = 1
            options["temperature"] = 0.0

        return options

    def _convert_to_canonical(
        self,
        whisper_result: dict[str, Any],
        duration_sec: float,
    ) -> CanonicalTranscript:
        """Convert whisper-timestamped result to canonical format.

        Args:
            whisper_result: Result from whisper_timestamped.transcribe().
            duration_sec: Video duration in seconds.

        Returns:
            CanonicalTranscript.
        """
        start_time = time.perf_counter()
        segments = []
        total_words = 0

        whisper_segments = whisper_result.get("segments", [])
        logger.debug(f"Converting {len(whisper_segments)} segments to canonical format")

        for seg in whisper_segments:
            words = None

            # Extract word-level timestamps if available
            if "words" in seg:
                word_list = []
                for w in seg["words"]:
                    text = w.get("text", "").strip()
                    if text:
                        # Use model_construct() to bypass Pydantic validation
                        # This is a significant performance optimization when processing
                        # thousands of word-level timestamps
                        word_obj = Word.model_construct(
                            start=w["start"],
                            end=w["end"],
                            word=text,
                            confidence=w.get("confidence"),
                        )
                        word_list.append(word_obj)
                        total_words += 1
                words = word_list if word_list else None

            # Use model_construct for Segment as well for consistency
            segment = Segment.model_construct(
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
                speaker=None,  # Whisper doesn't do diarization
                words=words,
            )
            segments.append(segment)

        # Use detected language or fall back to config
        language = whisper_result.get("language", self.config.engine.language)

        # Use actual duration from last segment if video probe failed
        if duration_sec <= 0 and segments:
            duration_sec = segments[-1].end

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        logger.debug(f"Conversion complete: {len(segments)} segments, "
                     f"{total_words} words in {elapsed_ms:.1f}ms")

        return CanonicalTranscript(
            engine="whisper",
            model=self.config.whisper.model,
            language=language,
            duration_sec=max(duration_sec, segments[-1].end if segments else 0),
            segments=segments,
        )
