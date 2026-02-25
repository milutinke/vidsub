"""Progress bar utilities for vidsub.

Provides optional progress bars via tqdm with graceful fallback
when tqdm is not installed or disabled.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

# Try to import tqdm, but provide fallback if not available
try:
    from tqdm import tqdm

    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    tqdm = None  # type: ignore[misc, assignment]


def is_progress_available() -> bool:
    """Check if progress bars are available (tqdm installed)."""
    return _TQDM_AVAILABLE


def get_progress_bar(
    iterable: Iterable[T] | None = None,
    total: int | None = None,
    desc: str | None = None,
    disable: bool = False,
    unit: str = "it",
    unit_scale: bool = False,
    **kwargs: Any,
) -> Iterable[T] | Any:
    """Get a progress bar wrapper for an iterable.

    This function provides a progress bar via tqdm when available,
    or returns the original iterable if tqdm is not installed
    or progress is disabled.

    Args:
        iterable: The iterable to wrap with a progress bar.
        total: Total number of items (for manually updated progress bars).
        desc: Description to show on the progress bar.
        disable: If True, disable the progress bar entirely.
        unit: Unit of iteration (default: "it" for items).
        unit_scale: If True, automatically scale units.
        **kwargs: Additional arguments passed to tqdm.

    Returns:
        Either a tqdm progress bar iterator or the original iterable
        if progress bars are unavailable or disabled.

    Example:
        >>> for item in get_progress_bar(items, desc="Processing"):
        ...     process(item)

        >>> pbar = get_progress_bar(total=100, desc="Working")
        >>> for i in range(100):
        ...     pbar.update(1)
        >>> pbar.close()
    """
    if disable or not _TQDM_AVAILABLE:
        if iterable is not None:
            return iterable
        # Return a simple manual progress tracker fallback
        return _ManualProgress(total, desc)

    # tqdm is available and enabled
    return tqdm(
        iterable=iterable,
        total=total,
        desc=desc,
        disable=False,
        unit=unit,
        unit_scale=unit_scale,
        **kwargs,
    )


class _ManualProgress:
    """Simple fallback progress tracker when tqdm is unavailable.

    This class provides a minimal progress tracking interface that
    logs progress at reasonable intervals without external dependencies.
    """

    def __init__(self, total: int | None = None, desc: str | None = None):
        """Initialize manual progress tracker.

        Args:
            total: Total number of items expected.
            desc: Description of the progress operation.
        """
        self.total = total
        self.desc = desc or "Progress"
        self.n = 0
        self._last_logged_percent = -1
        self._log_interval = 10  # Log every 10%

    def update(self, n: int = 1) -> None:
        """Update progress by n items.

        Args:
            n: Number of items completed since last update.
        """
        self.n += n

        if self.total and self.total > 0:
            percent = int((self.n / self.total) * 100)
            # Log at intervals and at completion
            if percent >= self._last_logged_percent + self._log_interval or percent >= 100:
                logger.info(f"{self.desc}: {percent}% ({self.n}/{self.total})")
                self._last_logged_percent = percent
        else:
            # No total known, log periodically
            if self.n % 100 == 0:
                logger.info(f"{self.desc}: {self.n} items processed")

    def close(self) -> None:
        """Close the progress tracker and log final status."""
        if self.total:
            logger.info(f"{self.desc}: complete ({self.n}/{self.total})")
        else:
            logger.info(f"{self.desc}: complete ({self.n} items)")

    def __enter__(self) -> _ManualProgress:
        """Context manager entry."""
        return self

    def __exit__(self, *args: Any) -> None:
        """Context manager exit."""
        self.close()


class ProgressManager:
    """Manager for progress bars that can be enabled/disabled globally."""

    _enabled: bool = True

    @classmethod
    def set_enabled(cls, enabled: bool) -> None:
        """Set whether progress bars are enabled globally.

        Args:
            enabled: If True, progress bars will be shown when available.
        """
        cls._enabled = enabled

    @classmethod
    def is_enabled(cls) -> bool:
        """Check if progress bars are enabled."""
        return cls._enabled and _TQDM_AVAILABLE

    @classmethod
    def progress(
        cls,
        iterable: Iterable[T] | None = None,
        total: int | None = None,
        desc: str | None = None,
        **kwargs: Any,
    ) -> Iterable[T] | Any:
        """Get a progress bar with global enable/disable check.

        Args:
            iterable: The iterable to wrap.
            total: Total number of items.
            desc: Description for the progress bar.
            **kwargs: Additional arguments for tqdm.

        Returns:
            Progress bar iterator or original iterable.
        """
        return get_progress_bar(
            iterable=iterable,
            total=total,
            desc=desc,
            disable=not cls.is_enabled(),
            **kwargs,
        )
