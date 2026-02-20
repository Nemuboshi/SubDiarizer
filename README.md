# SubDiarizer

A CLI tool that uses pyannote to diarize speakers and annotate .ass subtitles with speaker tags, making them easier for LLMs to use.

⚠️ **Speaker identification is not very accurate and can only produce coarse labels like `[SPEAKER_06]`. I still expect it to be useful for LLM translation.**

## Requirements

- Python `3.11`
- `uv`
- Optional CUDA GPU for faster diarization
- On Windows, FFmpeg **shared DLL** build is required for TorchCodec

## Windows FFmpeg Requirement

On Windows, you must install an FFmpeg build that ships shared DLLs (for example, a `full-shared` build).  
`torchcodec` needs DLLs such as `avcodec-*.dll` to be discoverable at runtime.

Quick check in `cmd`:

```bat
where avcodec-*.dll
```

If this command returns nothing, your FFmpeg DLLs are not discoverable yet.

## Install

```bash
uv sync
```

## Model Behavior (Offline-First)

- Local model path is derived automatically as:
  - `models/<repo_name>`
  - currently: `models/speaker-diarization-community-1`
- If downloading is needed, provide token via `--token`
  - or environment variable `HF_TOKEN` (also supports `hf_token`)

## Usage

```bash
uv run python main.py \
  --audio audio.aac \
  --ass input.ass \
  --output output.ass \
  --token <HF_TOKEN>
```

If model already exists locally, `--token` can be omitted.

## CLI Options

- `--audio` (required): input audio path
- `--ass` (required): input subtitle (`.ass`) path
- `--output` (required): output subtitle (`.ass`) path
- `--token` (optional): Hugging Face token for model download/access
- `--min_ratio` (default `0.3`): minimum overlap ratio to accept a speaker label (must be in `[0, 1]`)
- `--match_pad` (default `0.1`): seconds added to both subtitle boundaries when matching (must be `>= 0`)
- `--no_progress`: disable progress bars


## Matching Algorithm
For each subtitle line:
 - Build match window `[start - match_pad, end + match_pad]`
 - Find overlapping speaker segment with maximum overlap
 - Compute `ratio = overlap / subtitle_duration`
 - If `ratio >= min_ratio`, assign that speaker; otherwise assign `UNKNOWN`
