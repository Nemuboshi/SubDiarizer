import argparse
from datetime import datetime
import logging
import os
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, TypeVar, runtime_checkable

EPS = 1e-6
SHORT_LINE_THRESHOLD_SEC = 1.0
T = TypeVar("T")
MODEL_REPO_ID = "pyannote/speaker-diarization-community-1"
MODEL_DIR = Path("models") / MODEL_REPO_ID.rsplit("/", maxsplit=1)[-1]


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\x1b[36m",
        logging.INFO: "\x1b[32m",
        logging.WARNING: "\x1b[33m",
        logging.ERROR: "\x1b[31m",
        logging.CRITICAL: "\x1b[35m",
    }
    RESET = "\x1b[0m"

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        if not color:
            return message
        return f"{color}{message}{self.RESET}"


@runtime_checkable
class SupportsTimeBounds(Protocol):
    start: float
    end: float


@runtime_checkable
class SupportsItertracks(Protocol):
    def itertracks(self, *, yield_label: bool) -> Iterable[tuple[SupportsTimeBounds, object, object]]: ...


@runtime_checkable
class SupportsSubtitleLine(Protocol):
    start: int
    end: int
    text: str


@dataclass
class SpeakerSegment:
    start: float
    end: float
    speaker: str


def setup_logging(log_file: str) -> logging.Logger:
    logger = logging.getLogger("SubDiarizer")
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(ColorFormatter("%(message)s"))
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(file_handler)

    return logger


def with_progress(iterable: Iterable[T], enabled: bool, desc: str, total: int | None = None) -> Iterable[T]:
    if not enabled:
        return iterable
    try:
        from rich.progress import track
    except ImportError:
        return iterable
    return track(iterable, total=total, description=desc)


def setup_ffmpeg_dll_path() -> None:
    if sys.platform != "win32":
        return

    def _try_add_dll_dir(path: str | Path) -> bool:
        if not path:
            return False
        path_obj = Path(path)
        if not path_obj.is_dir():
            return False
        if not any(path_obj.glob("avcodec-*.dll")):
            return False
        os.add_dll_directory(str(path_obj))
        return True

    script_dir = Path(__file__).resolve().parent
    if _try_add_dll_dir(script_dir):
        return

    if _try_add_dll_dir(Path.cwd()):
        return

    try:
        where_result = subprocess.run(
            ["where.exe", "avcodec-*.dll"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return

    if where_result.returncode == 0:
        for line in where_result.stdout.splitlines():
            candidate = line.strip()
            if _try_add_dll_dir(Path(candidate).parent):
                return


def overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
    return max(0.0, min(a_end, b_end) - max(a_start, b_start))


def load_pipeline_with_fallback(model_dir: Path, token: str | None, logger: logging.Logger):
    from pyannote.audio import Pipeline

    model_dir.mkdir(parents=True, exist_ok=True)

    try:
        return Pipeline.from_pretrained(str(model_dir), token=token)
    except Exception as local_exc:
        logger.warning("Local model load failed. Falling back to snapshot download...")
        hf_http_error_type: type[BaseException] | None = None
        try:
            from huggingface_hub import snapshot_download
            from huggingface_hub.utils import HfHubHTTPError as ImportedHfHubHTTPError

            hf_http_error_type = ImportedHfHubHTTPError

            snapshot_download(
                repo_id=MODEL_REPO_ID,
                local_dir=model_dir,
                token=token,
            )
        except Exception as download_exc:
            download_hint = "Check network, token, and local filesystem permissions."
            if hf_http_error_type is not None and isinstance(download_exc, hf_http_error_type):
                response_obj = getattr(download_exc, "response", None)
                status_code = getattr(response_obj, "status_code", None)
                if status_code in (401, 403):
                    download_hint = "Authentication failed. Provide --token or HF_TOKEN with model access permission."
                elif status_code is not None:
                    download_hint = f"Hugging Face HTTP error: {status_code}."
            elif isinstance(download_exc, OSError):
                download_hint = "Local filesystem write failed while downloading model."

            msg = (
                "Local model is unavailable or incomplete, and auto-download failed.\n"
                f"Local load error: {local_exc}\n"
                f"Download error: {download_exc}\n"
                f"Model directory: {model_dir}\n"
                f"Hint: {download_hint}"
            )
            raise RuntimeError(msg) from download_exc

        try:
            return Pipeline.from_pretrained(str(model_dir), token=token)
        except Exception as retry_exc:
            msg = (
                "Model download completed, but loading still failed.\n"
                f"Retry load error: {retry_exc}\n"
                f"Model directory: {model_dir}"
            )
            raise RuntimeError(msg) from retry_exc


def resolve_diarization_output(result: object, logger: logging.Logger) -> object:
    exclusive = getattr(result, "exclusive_speaker_diarization", None)
    if exclusive is not None:
        logger.info("Using exclusive speaker diarization output.")
        return exclusive
    speaker_diarization = getattr(result, "speaker_diarization", None)
    if speaker_diarization is not None:
        logger.info("Using standard speaker diarization output.")
        return speaker_diarization
    logger.info("Using direct pipeline output as diarization result.")
    return result


def main(
    audio_path: str,
    ass_path: str,
    output_path: str,
    hf_token: str | None = None,
    min_ratio: float = 0.3,
    match_pad: float = 0.1,
    show_progress: bool = True,
) -> None:
    if not 0.0 <= min_ratio <= 1.0:
        msg = f"min_ratio must be in [0, 1], got {min_ratio}."
        raise ValueError(msg)
    if match_pad < 0.0:
        msg = f"match_pad must be >= 0, got {match_pad}."
        raise ValueError(msg)

    log_dir = Path(__file__).resolve().parent / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = str(log_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = setup_logging(log_file)
    setup_ffmpeg_dll_path()
    resolved_token = hf_token or os.environ.get("HF_TOKEN") or os.environ.get("hf_token")

    warnings.filterwarnings(
        "ignore",
        message=".*torchcodec is not installed correctly.*",
        category=UserWarning,
    )
    import pysubs2
    import torch
    import torchaudio

    logger.info("Loading diarization model...")
    pipeline = load_pipeline_with_fallback(MODEL_DIR, resolved_token, logger)
    if pipeline is None:
        msg = "Failed to load pyannote diarization pipeline."
        raise RuntimeError(msg)

    if torch.cuda.is_available():
        pipeline.to(torch.device("cuda"))
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("Using GPU")
    else:
        logger.info("Using CPU")

    logger.info("Loading audio waveform...")
    waveform, sample_rate = torchaudio.load(audio_path)

    logger.info("Running diarization...")
    result: object
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if show_progress:
            from pyannote.audio.pipelines.utils.hook import ProgressHook

            with ProgressHook() as hook:
                result = pipeline({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
        else:
            result = pipeline({"waveform": waveform, "sample_rate": sample_rate})
    diarization = resolve_diarization_output(result, logger)
    if diarization is None or not isinstance(diarization, SupportsItertracks):
        msg = "Unexpected diarization output from pipeline."
        raise RuntimeError(msg)

    speaker_segments: list[SpeakerSegment] = []
    for segment, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append(SpeakerSegment(start=segment.start, end=segment.end, speaker=str(speaker)))

    logger.info("Detected %d speaker segments", len(speaker_segments))

    logger.info("Loading subtitles...")
    subs = pysubs2.load(ass_path)

    logger.info("Matching speakers to subtitle lines...")
    speaker_segments.sort(key=lambda seg: (seg.start, seg.end))

    subtitle_items: list[tuple[int, SupportsSubtitleLine, float, float]] = []
    for idx, line in enumerate(subs):
        if not isinstance(line, SupportsSubtitleLine):
            msg = f"Unexpected subtitle line type at index {idx}."
            raise RuntimeError(msg)
        if not line.text.strip():
            continue
        line_start = line.start / 1000.0
        line_end = line.end / 1000.0
        subtitle_items.append((idx, line, line_start, line_end))

    subtitle_items.sort(key=lambda item: (item[2], item[3]))

    accepted_count = 0
    unknown_count = 0
    total_ratio = 0.0
    short_total = 0
    short_accepted = 0
    seg_cursor = 0

    for idx, line, line_start, line_end in with_progress(
        subtitle_items,
        enabled=show_progress,
        desc="Labeling subtitles",
        total=len(subtitle_items),
    ):
        line_duration = max(line_end - line_start, EPS)
        match_start = max(0.0, line_start - match_pad)
        match_end = line_end + match_pad

        while seg_cursor < len(speaker_segments) and speaker_segments[seg_cursor].end < match_start:
            seg_cursor += 1

        best_speaker = None
        best_overlap = 0.0
        scan_cursor = seg_cursor
        while scan_cursor < len(speaker_segments) and speaker_segments[scan_cursor].start <= match_end:
            seg = speaker_segments[scan_cursor]
            ov = overlap(match_start, match_end, seg.start, seg.end)
            if ov > best_overlap:
                best_overlap = ov
                best_speaker = seg.speaker
            scan_cursor += 1

        best_ratio = min(best_overlap / line_duration, 1.0)
        total_ratio += best_ratio

        is_short_line = line_duration <= SHORT_LINE_THRESHOLD_SEC
        if is_short_line:
            short_total += 1

        accepted = bool(best_speaker) and best_ratio >= min_ratio
        if accepted:
            line.text = f"[{best_speaker}] {line.text}"
            accepted_count += 1
            reason = "accepted_by_ratio"
            if is_short_line:
                short_accepted += 1
        else:
            line.text = f"[UNKNOWN] {line.text}"
            unknown_count += 1
            reason = "rejected_below_ratio"

        logger.debug(
            "line=%d start=%.3f end=%.3f match_start=%.3f match_end=%.3f duration=%.3f speaker=%s overlap=%.3f ratio=%.3f min_ratio=%.3f decision=%s",
            idx,
            line_start,
            line_end,
            match_start,
            match_end,
            line_duration,
            best_speaker if best_speaker else "NONE",
            best_overlap,
            best_ratio,
            min_ratio,
            reason,
        )

    total_labeled = accepted_count + unknown_count
    if total_labeled > 0:
        logger.info(
            "Matching summary: total=%d accepted=%d (%.1f%%) unknown=%d (%.1f%%) avg_ratio=%.3f",
            total_labeled,
            accepted_count,
            accepted_count * 100.0 / total_labeled,
            unknown_count,
            unknown_count * 100.0 / total_labeled,
            total_ratio / total_labeled,
        )
    else:
        logger.info("Matching summary: no non-empty subtitle lines to label.")

    if short_total > 0:
        logger.info(
            "Short-line summary (<=%.1fs): total=%d accepted=%d (%.1f%%)",
            SHORT_LINE_THRESHOLD_SEC,
            short_total,
            short_accepted,
            short_accepted * 100.0 / short_total,
        )

    logger.info("Saving output...")
    subs.save(output_path)
    logger.info("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--audio", required=True, help="Path to audio file")
    parser.add_argument("--ass", required=True, help="Path to input ASS file")
    parser.add_argument("--output", required=True, help="Path to output ASS file")
    parser.add_argument("--token", required=False, help="Hugging Face token (used for model download/access)")
    parser.add_argument(
        "--min_ratio",
        type=float,
        default=0.3,
        help="Minimum overlap ratio relative to subtitle duration",
    )
    parser.add_argument(
        "--match_pad",
        type=float,
        default=0.1,
        help="Padding (seconds) added to both subtitle boundaries when matching",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar",
    )

    args = parser.parse_args()
    main(
        args.audio,
        args.ass,
        args.output,
        args.token,
        args.min_ratio,
        args.match_pad,
        not args.no_progress,
    )
