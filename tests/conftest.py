"""pytest configuration and fixtures."""

import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Return path to test data directory."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture(scope="session")
def sample_video(test_data_dir: Path) -> Path:
    """Return path to sample video for testing.

    Creates a simple test video if it doesn't exist.
    """
    video_path = test_data_dir / "sample_video.mp4"

    if not video_path.exists():
        pytest.skip(
            "Sample video not found. Run: "
            "ffmpeg -f lavfi -i testsrc=duration=10:size=640x480:rate=30 "
            "-f lavfi -i sine=frequency=1000:duration=10 "
            "-pix_fmt yuv420p tests/fixtures/sample_video.mp4"
        )

    return video_path


@pytest.fixture
def temp_video(tmp_path: Path) -> Path:
    """Create a temporary video file for testing."""
    output = tmp_path / "test_video.mp4"

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-f", "lavfi",
                "-i", "testsrc=duration=1:size=320x240:rate=1",
                "-f", "lavfi",
                "-i", "anullsrc=r=44100:cl=mono",
                "-pix_fmt", "yuv420p",
                "-shortest",
                str(output),
            ],
            capture_output=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("FFmpeg not available for creating test video")

    return output
