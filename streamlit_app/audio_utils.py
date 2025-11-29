from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import librosa
import numpy as np
import soundfile as sf
import torch

SAMPLE_RATE = 16_000
N_MELS = 128
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
MIN_DURATION = 0.5
MAX_DURATION = 5.0


@dataclass(slots=True)
class AudioProcessingResult:
    waveform: np.ndarray
    duration: float
    logmel: np.ndarray
    frame_count: int


def _load_from_bytes(data: bytes, sr: int) -> np.ndarray:
    """Decode audio bytes into a mono waveform at the desired sample rate."""
    buffer = io.BytesIO(data)
    audio, orig_sr = librosa.load(buffer, sr=None, mono=True)
    if orig_sr != sr:
        audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    return audio.astype(np.float32)


def _load_from_path(path: Path, sr: int) -> np.ndarray:
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio.astype(np.float32)


def load_waveform(source: bytes | Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load raw waveform from bytes or path."""
    if isinstance(source, (bytes, bytearray)):
        return _load_from_bytes(bytes(source), sr)
    return _load_from_path(Path(source), sr)


def standardize_waveform(waveform: np.ndarray) -> Tuple[np.ndarray, float]:
    """Trim silence, peak-normalise, and validate duration."""
    waveform, _ = librosa.effects.trim(waveform, top_db=30)
    peak = np.max(np.abs(waveform))
    if peak > 0:
        waveform = waveform / peak
    duration = waveform.shape[0] / SAMPLE_RATE
    return waveform, duration


def waveform_to_logmel(waveform: np.ndarray) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=waveform,
        sr=SAMPLE_RATE,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )
    logmel = librosa.power_to_db(mel, ref=np.max)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-8)
    return logmel.astype(np.float32)


def build_features(source: bytes | Path) -> AudioProcessingResult:
    waveform = load_waveform(source)
    waveform, duration = standardize_waveform(waveform)
    if duration < MIN_DURATION or duration > MAX_DURATION:
        raise ValueError(f"Audio duration {duration:.2f}s di luar rentang {MIN_DURATION}-{MAX_DURATION}s")
    logmel = waveform_to_logmel(waveform)
    frame_count = int(logmel.shape[1])
    return AudioProcessingResult(waveform=waveform, duration=duration, logmel=logmel, frame_count=frame_count)


def logmel_to_tensor(logmel: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.from_numpy(logmel).unsqueeze(0).unsqueeze(0)
    lengths = torch.tensor([logmel.shape[1]], dtype=torch.long)
    return tensor, lengths


def save_waveform(path: Path, waveform: np.ndarray, sr: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, waveform, sr)


def list_recent_files(paths: Iterable[Path], limit: int = 10) -> list[str]:
    resolved = sorted((p for p in paths if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
    return [p.name for p in resolved[:limit]]
