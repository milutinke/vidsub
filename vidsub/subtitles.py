"""Subtitle format writers (SRT, ASS)."""

from __future__ import annotations

import io
import re
from pathlib import Path
from typing import TYPE_CHECKING

from vidsub.segmentation import Caption, segment_to_captions

if TYPE_CHECKING:
    from vidsub.models import AssStyleConfig, CanonicalTranscript, SubtitleConfig


def _format_srt_time(seconds: float) -> str:
    """Format time as SRT timestamp: HH:MM:SS,mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def _escape_srt_text(text: str) -> str:
    """Escape special characters for SRT.

    SRT uses basic HTML tags for formatting:
    - <b>, </b> for bold
    - <i>, </i> for italic
    - <u>, </u> for underline
    """
    # Remove ASS-style line breaks
    text = text.replace("\\N", "\n")
    # Basic HTML escaping
    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")
    return text


def generate_srt(
    transcript: CanonicalTranscript,
    config: SubtitleConfig,
) -> str:
    """Generate SRT subtitle content from transcript.

    Args:
        transcript: Canonical transcript.
        config: Subtitle configuration.

    Returns:
        SRT formatted string.
    """
    output = io.StringIO()
    caption_index = 1

    for segment in transcript.segments:
        captions = segment_to_captions(segment, config)

        for caption in captions:
            # Write caption number
            output.write(f"{caption_index}\n")

            # Write timing
            start_str = _format_srt_time(caption.start)
            end_str = _format_srt_time(caption.end)
            output.write(f"{start_str} --> {end_str}\n")

            # Write text (may be multi-line)
            for line in caption.lines:
                output.write(f"{_escape_srt_text(line)}\n")

            output.write("\n")
            caption_index += 1

    return output.getvalue()


def write_srt(
    transcript: CanonicalTranscript,
    config: SubtitleConfig,
    output_path: Path,
) -> Path:
    """Write SRT file.

    Args:
        transcript: Transcript to convert.
        config: Subtitle configuration.
        output_path: Output file path.

    Returns:
        Path to written file.
    """
    content = generate_srt(transcript, config)
    output_path.write_text(content, encoding="utf-8")
    return output_path


class SrtParser:
    """Parse SRT files for validation."""

    SRT_BLOCK_RE = re.compile(
        r"(\d+)\s+"
        r"(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})"
        r"(.*?)"
        r"(?=\n\d+\s+\d{2}:\d{2}:|\Z)",
        re.DOTALL,
    )

    @staticmethod
    def parse_time(time_str: str) -> float:
        """Parse SRT time string to seconds."""
        time_str = time_str.replace(",", ".")
        parts = time_str.split(":")
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    def parse(self, content: str) -> list[Caption]:
        """Parse SRT content into captions."""
        captions = []

        for match in self.SRT_BLOCK_RE.finditer(content):
            start_str = match.group(2)
            end_str = match.group(3)
            text = match.group(4).strip()

            start = self.parse_time(start_str)
            end = self.parse_time(end_str)

            # Split lines
            lines = text.split("\n")

            captions.append(Caption(start=start, end=end, lines=lines))

        return captions

    def parse_file(self, path: Path) -> list[Caption]:
        """Parse SRT file."""
        content = path.read_text(encoding="utf-8")
        return self.parse(content)


def _format_ass_time(seconds: float) -> str:
    """Format time as ASS timestamp: H:MM:SS.cc"""
    centis = int((seconds % 1) * 100)
    secs = int(seconds % 60)
    mins = int((seconds // 60) % 60)
    hours = int(seconds // 3600)
    return f"{hours}:{mins:02d}:{secs:02d}.{centis:02d}"


def _escape_ass_text(text: str) -> str:
    r"""Escape special characters for ASS.

    ASS uses:
    - \N for forced line break
    - \n for smart line break (wrapping)
    - {...} for override tags
    """
    # Escape backslashes first
    text = text.replace("\\", "\\\\")
    # Escape braces (used for override tags)
    text = text.replace("{", "\\{")
    text = text.replace("}", "\\}")
    return text


def _parse_color(color_str: str) -> str:
    """Parse color string to ASS format.

    Args:
        color_str: Color name or hex string.

    Returns:
        ASS color in &HAABBGGRR format.
    """
    # Named colors
    colors = {
        "black": "&H00000000",
        "white": "&H00FFFFFF",
        "red": "&H000000FF",
        "green": "&H0000FF00",
        "blue": "&H00FF0000",
        "yellow": "&H0000FFFF",
        "cyan": "&H00FFFF00",
        "magenta": "&H00FF00FF",
        "transparent": "&H00000000",
    }

    if color_str.lower() in colors:
        return colors[color_str.lower()]

    # Hex color parsing
    hex_str = color_str.lstrip("#")
    if len(hex_str) == 6:
        # RGB to BGR
        r = hex_str[0:2]
        g = hex_str[2:4]
        b = hex_str[4:6]
        return f"&H00{b}{g}{r}"

    # Default to transparent
    return "&H00000000"


def generate_ass_header(
    config: SubtitleConfig,
    style_config: AssStyleConfig,
) -> str:
    """Generate ASS header with styles.

    Args:
        config: Subtitle configuration.
        style_config: ASS styling configuration.

    Returns:
        ASS header string.
    """
    # ASS color format: &HAABBGGRR (alpha, blue, green, red)
    # For black background with no transparency: &H00000000
    # For transparent: &H00000000 with special handling

    bg_style = config.background_style

    # Base style
    header = """[Script Info]
Title: vidsub generated subtitles
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
"""

    # Determine style based on background
    if bg_style == "solid":
        # Solid background - use BorderStyle 3 (opaque box)
        # BackColour controls the box color
        bg_colour = _parse_color(style_config.bg_color)
        style_line = (
            f"Style: Default,{style_config.font_name},{style_config.font_size},"
            f"&H00FFFFFF,&H000000FF,&H00000000,{bg_colour},"
            f"0,0,0,0,100,100,0,0,3,{style_config.outline},0,2,10,10,{style_config.margin_v},1"
        )
    else:
        # No background - use BorderStyle 1 (outline)
        style_line = (
            f"Style: Default,{style_config.font_name},{style_config.font_size},"
            f"&H00FFFFFF,&H000000FF,&H00000000,&H00000000,"
            f"0,0,0,0,100,100,0,0,1,{style_config.outline},{style_config.shadow},2,10,10,{style_config.margin_v},1"
        )

    header += style_line + "\n\n"
    header += "[Events]\n"
    header += "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"

    return header


def generate_ass(
    transcript: CanonicalTranscript,
    config: SubtitleConfig,
    style_config: AssStyleConfig,
) -> str:
    """Generate ASS subtitle content from transcript.

    Args:
        transcript: Canonical transcript.
        config: Subtitle configuration.
        style_config: ASS styling configuration.

    Returns:
        ASS formatted string.
    """
    output = io.StringIO()

    # Write header
    output.write(generate_ass_header(config, style_config))

    # Write dialogue events
    for segment in transcript.segments:
        captions = segment_to_captions(segment, config)

        for caption in captions:
            start_str = _format_ass_time(caption.start)
            end_str = _format_ass_time(caption.end)

            # Join lines with ASS line break
            text = "\\N".join(_escape_ass_text(line) for line in caption.lines)

            # Dialogue line format
            dialogue = (
                f"Dialogue: 0,{start_str},{end_str},Default,,0,0,0,,{text}\n"
            )
            output.write(dialogue)

    return output.getvalue()


def write_ass(
    transcript: CanonicalTranscript,
    config: SubtitleConfig,
    style_config: AssStyleConfig,
    output_path: Path,
) -> Path:
    """Write ASS file.

    Args:
        transcript: Transcript to convert.
        config: Subtitle configuration.
        style_config: ASS styling configuration.
        output_path: Output file path.

    Returns:
        Path to written file.
    """
    content = generate_ass(transcript, config, style_config)
    output_path.write_text(content, encoding="utf-8")
    return output_path


class AssParser:
    """Parse ASS files for validation."""

    DIALOGUE_RE = re.compile(
        r"^Dialogue:\s*\d+,\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*[^,]*,\s*\d+,\s*\d+,\s*\d+,\s*[^,]*,\s*(.*)$"
    )

    @staticmethod
    def parse_time(time_str: str) -> float:
        """Parse ASS time string to seconds."""
        parts = time_str.split(":")
        hours = float(parts[0])
        minutes = float(parts[1])
        seconds = float(parts[2])
        return hours * 3600 + minutes * 60 + seconds

    def parse(self, content: str) -> list[Caption]:
        """Parse ASS content into captions."""
        captions = []
        in_events = False

        for line in content.split("\n"):
            line = line.strip()

            if line == "[Events]":
                in_events = True
                continue

            if in_events and line.startswith("["):
                break

            if in_events and line.startswith("Dialogue:"):
                match = self.DIALOGUE_RE.match(line)
                if match:
                    start_str = match.group(1)
                    end_str = match.group(2)
                    text = match.group(4)

                    start = self.parse_time(start_str)
                    end = self.parse_time(end_str)

                    # Unescape text
                    text = text.replace("\\N", "\n")
                    text = text.replace("\\n", " ")
                    text = text.replace("\\{", "{")
                    text = text.replace("\\}", "}")

                    lines = text.split("\n")
                    captions.append(Caption(start=start, end=end, lines=lines))

        return captions

    def parse_file(self, path: Path) -> list[Caption]:
        """Parse ASS file."""
        content = path.read_text(encoding="utf-8")
        return self.parse(content)
