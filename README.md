# vidsub

CLI tool for video subtitling with dual-engine transcription (Google Gemini or whisper-timestamped).

## Features

- **Dual transcription engines**: Choose between local Whisper or cloud Gemini
- **Subtitle formats**: SRT and ASS output
- **Subtitle burn-in**: Embed subtitles directly into video with FFmpeg
- **Configurable styling**: Font, size, background options, and ASS styling
- **FFmpeg presets**: Built-in presets for YouTube, web, and archival
- **Progress tracking**: Visual progress bars for long operations
- **Parallel Gemini chunking**: Bounded parallel chunk processing with retry/backoff
- **Cross-platform**: Linux, macOS, Windows

## Recommendation

Whisper is the recommended engine for most users. It generally provides better timestamp
accuracy, and it runs on your own hardware, so there are no API usage costs.

## Installation

### Requirements

- Python 3.14+
- FFmpeg 4.4+ (required for video processing)
- [uv](https://docs.astral.sh/uv/) (required for running the application)

**Install uv:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
For other installation methods, see [https://docs.astral.sh/uv/getting-started/installation/](https://docs.astral.sh/uv/getting-started/installation/)

### Setup

This application is designed to be run with `uv`. Use `uv run` to execute commands:

```bash
uv run vidsub --help
```

Or sync dependencies and run directly:

```bash
uv sync
uv run vidsub --help
```

## Quick Start

### 1. Initialize configuration

```bash
uv run vidsub init-config
```

This creates a `vidsub.yaml` configuration file with sensible defaults.

### 2. Set up API key (for Gemini engine)

```bash
export GEMINI_API_KEY="your-api-key"
```

Get an API key from [Google AI Studio](https://aistudio.google.com/).

### 3. Transcribe a video

```bash
# Using Whisper (local)
uv run vidsub run video.mp4 --engine whisper

# Using a custom Whisper model from Hugging Face
uv run vidsub run video.mp4 --engine whisper \
  --whisper-model sam8000/whisper-large-v3-turbo-serbian-serbia

# Using Gemini (cloud)
uv run vidsub run video.mp4 --engine gemini
```

### 4. Burn subtitles into video

```bash
# Burn with default settings
uv run vidsub run video.mp4 --engine whisper --burn

# Use a specific FFmpeg preset
uv run vidsub run video.mp4 --engine whisper --burn --preset youtube_1080p
```

## Commands

### `vidsub run`

Run the complete pipeline: transcribe → generate subtitles → optionally burn.

```bash
vidsub run <video> [options]

Options:
  --engine [whisper|gemini]     Transcription engine
  --whisper-model TEXT          Whisper model alias, local path, or Hugging Face repo id
  -c, --config PATH            Config file path
  -o, --out-dir PATH           Output directory
  -t, --temp-dir PATH          Temporary directory
  -l, --language CODE          Language code (e.g., en, es, fr)
  -f, --format [srt|ass|both]  Output format (default: both)
  --burn / --no-burn           Burn subtitles into video
  --subtitle-bg [none|solid]   Subtitle background style
  --preset TEXT                FFmpeg encoding preset
  --progress / --no-progress   Show progress bars
  --verbose-segmentation       Verbose segmentation logging
  --overwrite                  Overwrite existing files
  --keep-temp                  Keep temporary files
  --log-level [debug|info|warning|error]
```

### `vidsub transcribe`

Transcribe video and generate subtitle files only (no burn).

```bash
vidsub transcribe <video> [options]
```

Options are the same as `vidsub run` except `--burn` and `--preset` are not available.

### `vidsub burn`

Burn existing subtitle file into video.

```bash
vidsub burn <video> --subs <subtitle.ass> [options]

Options:
  -s, --subs PATH              Subtitle file path (required)
  -c, --config PATH            Config file path
  -o, --out-dir PATH           Output directory
  --subtitle-bg [none|solid]   Subtitle background style
  --preset TEXT                FFmpeg encoding preset
  --overwrite                  Overwrite existing files
  --log-level [debug|info|warning|error]
```

### `vidsub init-config`

Create a configuration file template.

```bash
vidsub init-config [--path PATH]
```

Default path is `vidsub.yaml` in the current directory.

### `vidsub validate`

Validate a transcript or subtitle file.

```bash
vidsub validate <file>
```

Supports:
- `.json` - Canonical transcript format
- `.srt` - SubRip subtitle format
- `.ass` - Advanced SubStation Alpha format

## Configuration

vidsub uses a YAML configuration file. It searches for config in the following order:

1. Path specified via `--config`
2. `vidsub.yaml` or `vidsub.yml` in current directory
3. `config/vidsub.yaml` in current directory
4. `.vidsub/config.yaml` in current directory

### Configuration Options

```yaml
app:
  out_dir: "./out"              # Output directory for all files
  temp_dir: "./.vidsub_tmp"     # Temporary directory
  overwrite: false              # Overwrite existing files
  keep_temp: false              # Keep temporary files after run
  burn: false                   # Burn subtitles by default
  show_progress: true           # Show progress bars
  verbose_postprocessing: false # Verbose segmentation logs

engine:
  name: "whisper"               # whisper | gemini
  language: "en"                # ISO 639-1 language code

whisper:
  # Built-in aliases: tiny, base, small, medium, large, large-v2, large-v3
  # Or use a local path / Hugging Face repo id
  model: "large"
  device: "auto"                # auto | cpu | cuda | mps
  vad: true                     # Voice activity detection
  accurate: true                # Accurate timestamp mode

gemini:
  model: "gemini-2.5-pro"       # Gemini model name
  api_key_env: "GEMINI_API_KEY" # Environment variable for API key
  chunk_seconds: 60             # Video chunk size (10-180)
  overlap_seconds: 2            # Chunk overlap (0-10)
  fps: 1                        # Frames per second sampling (1-10)
  max_retries: 2                # Retry attempts for failed chunks (0-5)
  concurrency: 3                # Concurrent chunk processing (1-8)
  upload_timeout_sec: 180       # Max wait for Gemini Files to become ACTIVE
  poll_interval_sec: 2.0        # Poll interval while waiting on file processing
  retry_base_delay_sec: 1.0     # Base delay for exponential backoff
  retry_max_delay_sec: 8.0      # Max delay cap for Gemini retries

subtitles:
  formats: ["srt", "ass"]       # Output formats
  max_chars_per_line: 42        # Max characters per line (20-100)
  max_lines: 2                  # Max lines per caption (1-3)
  max_caption_seconds: 6.0      # Max caption duration in seconds (1.0-20.0)
  split_on_silence_ms: 350      # Silence threshold for splitting (0-1000)
  background_style: "none"      # none | solid

style_ass:
  font_name: "Inter"            # Font family
  font_size: 44                 # Font size (10-200)
  outline: 3                    # Outline width (0-10)
  shadow: 0                     # Shadow depth (0-10)
  margin_v: 40                  # Vertical margin (0-500)
  bg_color: "black"             # Background color (when background_style=solid)

ffmpeg:
  default_preset: "youtube_1080p"
  presets:
    youtube_1080p:
      video_codec: "libx264"
      audio_codec: "copy"
      crf: 18
      preset: "medium"
      pixel_format: "yuv420p"
      description: "Optimized for YouTube 1080p uploads"

    youtube_4k:
      video_codec: "libx264"
      audio_codec: "copy"
      crf: 17
      preset: "slow"
      pixel_format: "yuv420p"
      description: "High quality for YouTube 4K uploads"

    web_optimized:
      video_codec: "libx264"
      audio_codec: "aac"
      audio_bitrate: "128k"
      crf: 23
      preset: "fast"
      pixel_format: "yuv420p"
      description: "Smaller file size for web sharing"

    archive:
      video_codec: "libx264"
      audio_codec: "flac"
      crf: 15
      preset: "veryslow"
      pixel_format: "yuv420p"
      description: "Maximum quality for archival"
```

`whisper.model` accepts:
- built-in aliases: `tiny`, `base`, `small`, `medium`, `large`, `large-v2`, `large-v3`
- local model paths such as `./models/whisper-large-v3-serbian`
- public Hugging Face repo IDs such as `sam8000/whisper-large-v3-turbo-serbian-serbia`

When you use a Hugging Face repo ID, vidsub downloads the snapshot first with the Hugging Face
library and then loads the local downloaded files for transcription. Public models do not require
login.

### Environment Variables

You can override config values using environment variables:

- `GEMINI_API_KEY` - API key for Gemini engine
- `VIDSUB_OUT_DIR` - Output directory
- `VIDSUB_TEMP_DIR` - Temporary directory
- `VIDSUB_ENGINE` - Default engine (whisper/gemini)

## FFmpeg Presets

vidsub includes built-in FFmpeg presets for common use cases:

| Preset | CRF | Preset | Use Case |
|--------|-----|--------|----------|
| `youtube_1080p` | 18 | medium | YouTube 1080p uploads |
| `youtube_4k` | 17 | slow | YouTube 4K uploads |
| `web_optimized` | 23 | fast | Web sharing (smaller files) |
| `archive` | 15 | veryslow | Maximum quality archival |

Use `--preset <name>` to select a preset, or define custom presets in your config file.

## Engine Comparison

| Feature | Whisper | Gemini |
|---------|---------|--------|
| Processing | Local | Cloud |
| Word-level timestamps | Yes | No |
| Accuracy | Excellent | Very Good |
| Speed | GPU-dependent | Network-dependent |
| Privacy | Keeps data local | Sends to Google |
| Setup | Model download (~3GB) | API key required |
| Cost | Free (local compute) | Per-token pricing |

## Setup Guide

### Development Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd vidsub
   ```

2. **Install dependencies with uv**
   ```bash
   uv sync
   ```

3. **Install FFmpeg** (if not already installed)

   **Ubuntu/Debian:**
   ```bash
   sudo apt update
   sudo apt install ffmpeg
   ```

   **macOS:**
   ```bash
   brew install ffmpeg
   ```

   **Note:** The default Homebrew FFmpeg does not include libass support needed for burning subtitles. To burn subtitles, install the full version:
   ```bash
   brew tap homebrew-ffmpeg/ffmpeg
   brew install homebrew-ffmpeg/ffmpeg/ffmpeg
   ```

   **Windows:**
   Download from [ffmpeg.org](https://ffmpeg.org/download.html) and add to PATH.

4. **Verify installation**
   ```bash
   uv run vidsub --help
   ffmpeg -version
   ```

5. **Run tests**
   ```bash
   uv run pytest
   ```

### First Run Setup

1. **Initialize configuration**
   ```bash
   uv run vidsub init-config
   ```

2. **For Whisper (local transcription):**
   - Built-in Whisper aliases download on first use (~3GB for `large`)
   - Hugging Face repo IDs download to the Hugging Face cache first, then run locally
   - Public Hugging Face models do not require login

3. **For Gemini (cloud transcription):**
   - Get an API key from [Google AI Studio](https://aistudio.google.com/)
   - Set the environment variable: `export GEMINI_API_KEY="your-key"`

4. **Test with a short video**
   ```bash
   uv run vidsub run test_video.mp4 --engine whisper --format srt
   ```

## Output Files

After running vidsub, the output directory will contain:

| File | Description |
|------|-------------|
| `transcript.json` | Canonical transcript with segments and word-level timing |
| `subtitles.srt` | SubRip subtitle file |
| `subtitles.ass` | Advanced SubStation Alpha file (with styling) |
| `*_burned.mp4` | Video with burned-in subtitles (if `--burn` used) |
| `run_report.json` | Processing statistics and settings snapshot |

## Troubleshooting

### FFmpeg not found
Ensure FFmpeg is installed and in your PATH:
```bash
which ffmpeg  # Linux/macOS
where ffmpeg  # Windows
```

### Whisper model download fails
Models are downloaded automatically on first use. If download fails:
```bash
# Manual download using whisper CLI
whisper --model large --download-only
```

For public Hugging Face models, pre-download the snapshot and retry:
```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('sam8000/whisper-large-v3-turbo-serbian-serbia', token=None)"
```

### Gemini API errors
- Verify `GEMINI_API_KEY` is set correctly
- Check your API quota at [Google AI Studio](https://aistudio.google.com/)
- For large videos, try reducing `chunk_seconds` in config

### Out of memory with Whisper
- Use a smaller model (e.g., `medium` instead of `large`)
- Enable VAD: `vad: true` in config
- Ensure `accurate: false` for lower memory usage

### FFmpeg does not have the 'subtitles' filter
If you see this error when burning subtitles, your FFmpeg was built without libass support:
```
FFmpeg does not have the 'subtitles' filter. This usually means FFmpeg was built without libass support.
```

**macOS fix:**
```bash
brew tap homebrew-ffmpeg/ffmpeg
brew install homebrew-ffmpeg/ffmpeg/ffmpeg
```

**Ubuntu/Debian fix:**
```bash
sudo apt update
sudo apt install ffmpeg libavfilter-extra
```

## License

MIT
