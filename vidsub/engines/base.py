"""Base interface for transcription engines."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vidsub.models import CanonicalTranscript, Config


class TranscriptionEngine(ABC):
    """Abstract base class for transcription engines.

    All engines must implement the transcribe method that takes
    an input path and returns a CanonicalTranscript.
    """

    def __init__(self, config: Config):
        """Initialize engine with configuration.

        Args:
            config: Application configuration.
        """
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        """Return engine name identifier."""
        pass

    @abstractmethod
    def transcribe(self, input_path: Path) -> CanonicalTranscript:
        """Transcribe audio/video file to canonical transcript.

        Args:
            input_path: Path to audio or video file.

        Returns:
            CanonicalTranscript with segments and optional word-level timing.

        Raises:
            FileNotFoundError: If input file doesn't exist.
            TranscriptionError: If transcription fails.
        """
        pass

    def validate_input(self, input_path: Path) -> None:
        """Validate input file exists.

        Args:
            input_path: Path to validate.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")


class TranscriptionError(Exception):
    """Exception raised when transcription fails."""
    pass


class EngineFactory:
    """Factory for creating transcription engine instances."""

    @staticmethod
    def create(config: Config) -> TranscriptionEngine:
        """Create appropriate engine based on config.

        Args:
            config: Application configuration.

        Returns:
            Configured TranscriptionEngine instance.

        Raises:
            ValueError: If engine name is unknown.
        """
        engine_name = config.engine.name.lower()

        if engine_name == "whisper":
            from vidsub.engines.whisper_engine import WhisperEngine

            return WhisperEngine(config)
        elif engine_name == "gemini":
            from vidsub.engines.gemini_engine import GeminiEngine

            return GeminiEngine(config)
        else:
            raise ValueError(f"Unknown engine: {engine_name}")
