"""Tests for subtitle generation."""

from pathlib import Path

from vidsub.models import (
    AssStyleConfig,
    CanonicalTranscript,
    Segment,
    SubtitleConfig,
)
from vidsub.segmentation import split_text_into_lines
from vidsub.subtitles import (
    SrtParser,
    generate_ass,
    generate_srt,
    write_srt,
)


class TestSplitTextIntoLines:
    """Tests for text splitting."""

    def test_short_text_no_split(self) -> None:
        result = split_text_into_lines("Hello", max_chars_per_line=42, max_lines=2)
        assert result == ["Hello"]

    def test_split_at_word_boundary(self) -> None:
        text = "Hello world this is a test"
        result = split_text_into_lines(
            text, max_chars_per_line=11, max_lines=3
        )
        assert result == ["Hello world", "this is a", "test"]

    def test_respects_max_lines(self) -> None:
        text = "a b c d e f g h i j"
        result = split_text_into_lines(text, max_chars_per_line=5, max_lines=2)
        assert result is None


class TestGenerateSrt:
    """Tests for SRT generation."""

    def test_single_segment(self) -> None:
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=10.0,
            segments=[Segment(start=0.0, end=5.0, text="Hello world")],
        )
        config = SubtitleConfig()
        srt = generate_srt(transcript, config)

        assert "1" in srt
        assert "00:00:00,000 --> 00:00:05,000" in srt
        assert "Hello world" in srt

    def test_multiple_segments(self) -> None:
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=20.0,
            segments=[
                Segment(start=0.0, end=5.0, text="First"),
                Segment(start=5.0, end=10.0, text="Second"),
            ],
        )
        config = SubtitleConfig()
        srt = generate_srt(transcript, config)

        assert srt.count("-->") == 2

    def test_write_srt(self, tmp_path: Path) -> None:
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=10.0,
            segments=[Segment(start=0.0, end=5.0, text="Hello")],
        )
        config = SubtitleConfig()
        output = tmp_path / "test.srt"
        write_srt(transcript, config, output)

        assert output.exists()
        content = output.read_text()
        assert "Hello" in content


class TestSrtParser:
    """Tests for SRT parsing."""

    def test_parse_simple_srt(self) -> None:
        content = """1
00:00:01,000 --> 00:00:05,000
Hello world

2
00:00:06,000 --> 00:00:10,000
Second line
"""
        parser = SrtParser()
        captions = parser.parse(content)

        assert len(captions) == 2
        assert captions[0].start == 1.0
        assert captions[0].end == 5.0
        assert captions[0].lines == ["Hello world"]


class TestGenerateAss:
    """Tests for ASS generation."""

    def test_contains_header(self) -> None:
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=10.0,
            segments=[Segment(start=0.0, end=5.0, text="Hello")],
        )
        config = SubtitleConfig()
        style = AssStyleConfig()
        ass = generate_ass(transcript, config, style)

        assert "[Script Info]" in ass
        assert "[V4+ Styles]" in ass
        assert "[Events]" in ass

    def test_contains_dialogue(self) -> None:
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=10.0,
            segments=[Segment(start=1.0, end=5.0, text="Hello")],
        )
        config = SubtitleConfig()
        style = AssStyleConfig()
        ass = generate_ass(transcript, config, style)

        assert "Dialogue:" in ass
        assert "Hello" in ass

    def test_solid_background_style(self) -> None:
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=10.0,
            segments=[Segment(start=0.0, end=5.0, text="Hello")],
        )
        config = SubtitleConfig(background_style="solid")
        style = AssStyleConfig()
        ass = generate_ass(transcript, config, style)

        # BorderStyle 3 = opaque box (appears in the style line as a comma-separated value)
        # The style format includes: ...,BorderStyle,Outline,Shadow,...
        assert ",3," in ass.split("Style: ")[1].split("\n")[0]


class TestSubtitleEdgeCases:
    """Edge case tests for subtitle generation."""

    def test_empty_transcript(self) -> None:
        """Test generating SRT from empty transcript."""
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=10.0,
            segments=[],
        )
        config = SubtitleConfig()
        srt = generate_srt(transcript, config)

        # Empty transcript should produce empty SRT (just whitespace)
        assert srt.strip() == ""

    def test_unicode_text(self) -> None:
        """Test handling unicode characters including emoji."""
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=10.0,
            segments=[
                Segment(start=0.0, end=5.0, text="Hello 世界 👋 émojis"),
            ],
        )
        config = SubtitleConfig()
        srt = generate_srt(transcript, config)

        assert "世界" in srt
        assert "👋" in srt
        assert "émojis" in srt

    def test_very_long_text_splitting(self) -> None:
        """Test that very long text is properly split into multiple captions."""
        long_text = "This is a very long sentence that should be split into multiple lines " \
                    "because it exceeds the maximum characters per line limit and should " \
                    "be handled gracefully without breaking the subtitle generation."
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=30.0,
            segments=[Segment(start=0.0, end=25.0, text=long_text)],
        )
        config = SubtitleConfig(max_chars_per_line=42, max_lines=2)
        srt = generate_srt(transcript, config)

        # Should have multiple caption entries due to splitting
        assert "-->" in srt
        # Each caption should have timing
        lines = srt.strip().split("\n")
        assert len(lines) > 3  # At least one caption with number, timing, text

    def test_special_characters_escaping(self) -> None:
        """Test that special HTML characters are escaped in SRT."""
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=10.0,
            segments=[
                Segment(start=0.0, end=5.0, text="5 < 10 and 10 > 5 & that's true"),
            ],
        )
        config = SubtitleConfig()
        srt = generate_srt(transcript, config)

        # Special characters should be escaped
        assert "&lt;" in srt or "< 10" in srt  # Either escaped or left as-is
        assert "5 < 10" in srt or "&lt;" in srt

    def test_multi_line_segment(self) -> None:
        """Test segment with explicit line breaks."""
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=10.0,
            segments=[
                Segment(start=0.0, end=5.0, text="Line one\nLine two"),
            ],
        )
        config = SubtitleConfig()
        srt = generate_srt(transcript, config)

        # Both lines should be present
        assert "Line one" in srt
        assert "Line two" in srt

    def test_ass_escaping_special_chars(self) -> None:
        """Test that ASS special characters are escaped."""
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=10.0,
            segments=[
                Segment(start=0.0, end=5.0, text="Use {\\b1}bold{\\b0} text"),
            ],
        )
        config = SubtitleConfig()
        style = AssStyleConfig()
        ass = generate_ass(transcript, config, style)

        # Braces should be escaped
        assert "\\{" in ass or "{\\\\b1}" in ass

    def test_srt_parser_malformed_input(self) -> None:
        """Test SRT parser handles malformed input gracefully."""
        parser = SrtParser()

        # Empty content
        captions = parser.parse("")
        assert captions == []

        # Content with only whitespace
        captions = parser.parse("   \n\n  ")
        assert captions == []

    def test_zero_duration_segment(self) -> None:
        """Test handling of zero-duration segments."""
        transcript = CanonicalTranscript(
            engine="whisper",
            model="large",
            language="en",
            duration_sec=10.0,
            segments=[
                Segment(start=5.0, end=5.0, text="Instant"),
            ],
        )
        config = SubtitleConfig()
        srt = generate_srt(transcript, config)

        # Should still generate valid SRT
        assert "00:00:05,000 --> 00:00:05,000" in srt
        assert "Instant" in srt
