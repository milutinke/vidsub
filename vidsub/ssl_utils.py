"""SSL certificate utilities for handling model downloads."""

from __future__ import annotations

import functools
import importlib
import logging
import ssl
import sys
import urllib.request
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any, cast

try:
    import certifi

    HAS_CERTIFI = True
except ImportError:
    HAS_CERTIFI = False

logger = logging.getLogger(__name__)


def _create_ssl_context_with_certifi() -> ssl.SSLContext | None:
    """Create an SSL context using certifi certificates.

    Returns:
        SSLContext configured with certifi certificates, or None if certifi
        is not installed.
    """
    if not HAS_CERTIFI:
        return None

    context = ssl.create_default_context()
    context.load_verify_locations(certifi.where())
    return context


@contextmanager
def patched_urlopen() -> Generator[None]:
    """Temporarily patch urllib.request.urlopen to use certifi certificates.

    This context manager monkey-patches urllib.request.urlopen to inject
    an SSL context that uses the certifi certificate bundle. This is necessary
    on macOS and some Linux distributions where Python's default SSL context
    cannot verify certificates from popular CDNs (like OpenAI's model hosting).

    Also patches torch.hub's imported urlopen reference if torch is loaded,
    since torch.hub imports urlopen directly at module load time.

    Yields:
        None

    Example:
        with patched_urlopen():
            # HTTP requests here will use certifi certificates
            model = whisper.load_model("large")
    """
    if not HAS_CERTIFI:
        logger.warning("certifi not installed; SSL certificate issues may occur")
        yield
        return

    original_urlopen = urllib.request.urlopen

    @functools.wraps(original_urlopen)
    def urlopen_with_certifi(
        url: str | urllib.request.Request,
        data: bytes | None = None,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> Any:
        """Wrapper that injects certifi SSL context."""
        # Only inject context if not already provided and it's an HTTPS URL
        if "context" not in kwargs:
            url_str = url if isinstance(url, str) else url.full_url
            if url_str.startswith("https://"):
                kwargs["context"] = _create_ssl_context_with_certifi()
                logger.debug("Using certifi certificates for %s", url_str)
        return original_urlopen(url, data, timeout, **kwargs)

    # Patch urllib.request.urlopen
    urllib.request.urlopen = cast(Any, urlopen_with_certifi)

    # Also patch torch.hub's imported urlopen if torch is already loaded
    # torch.hub does: from urllib.request import Request, urlopen
    # So we need to patch its module-level reference too
    torch_hub_patched = False
    if "torch" in sys.modules:
        try:
            torch_hub = cast(Any, importlib.import_module("torch.hub"))

            if hasattr(torch_hub, "urlopen"):
                torch_hub.urlopen = urlopen_with_certifi
                torch_hub_patched = True
                logger.debug("Patched torch.hub.urlopen")
        except Exception as e:
            logger.debug("Could not patch torch.hub.urlopen: %s", e)

    try:
        yield
    finally:
        urllib.request.urlopen = cast(Any, original_urlopen)
        if torch_hub_patched:
            try:
                torch_hub = cast(Any, importlib.import_module("torch.hub"))

                torch_hub.urlopen = original_urlopen
            except Exception:
                pass


@contextmanager
def patched_ssl_context() -> Generator[None]:
    """Temporarily patch SSL context creation to use certifi certificates.

    This context manager monkey-patches ssl.create_default_context() to use
    the certifi certificate bundle. This is used as a fallback for libraries
    that create their own SSL contexts.

    Yields:
        None
    """
    if not HAS_CERTIFI:
        logger.warning("certifi not installed; SSL certificate issues may occur")
        yield
        return

    original_create_default_context = ssl.create_default_context

    def create_context_with_certifi(*args: Any, **kwargs: Any) -> ssl.SSLContext:
        """Create SSL context using certifi certificate bundle."""
        context = original_create_default_context(*args, **kwargs)
        context.load_verify_locations(certifi.where())
        logger.debug("Using certifi certificate bundle: %s", certifi.where())
        return context

    ssl.create_default_context = create_context_with_certifi
    try:
        yield
    finally:
        ssl.create_default_context = original_create_default_context
