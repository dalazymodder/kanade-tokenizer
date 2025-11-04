import csv
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset

from ..util import _load_audio_internal, get_logger

logger = get_logger()


@dataclass
class AudioItem:
    waveform: torch.Tensor
    audio_id: str
    path: Path
    sample_rate: int
    frame_offset: int | None = None  # For chunked audio


def convert_to_mono(waveform: torch.Tensor) -> torch.Tensor:
    # (1, samples)
    if waveform.shape[0] > 1:
        return torch.mean(waveform, dim=0, keepdim=True)
    return waveform


def resample_audio(waveform: torch.Tensor, orig_freq: int, new_freq: int) -> torch.Tensor:
    if orig_freq != new_freq:
        resampler = torchaudio.transforms.Resample(orig_freq=orig_freq, new_freq=new_freq)
        return resampler(waveform)
    return waveform


def normalize_audio(waveform: torch.Tensor) -> torch.Tensor:
    max_val = torch.max(torch.abs(waveform)) + 1e-8
    return waveform / max_val


def preprocess_audio(
    waveform: torch.Tensor, sample_rate: int, mono: bool, normalize: bool, target_sample_rate: int | None = None
) -> tuple[torch.Tensor, int]:
    # Convert to mono if needed
    if mono:
        waveform = convert_to_mono(waveform)

    # Resample if needed
    if target_sample_rate is not None and sample_rate != target_sample_rate:
        waveform = resample_audio(waveform, sample_rate, target_sample_rate)
        sample_rate = target_sample_rate

    # Normalize if needed
    if normalize:
        waveform = normalize_audio(waveform)

    return waveform, sample_rate


def pad_audio(waveform: torch.Tensor, target_length: int) -> torch.Tensor:
    current_length = waveform.shape[1]
    if current_length >= target_length:
        return waveform

    # Calculate padding needed
    pad_length = target_length - current_length
    # Pad with zeros at the end
    padding = torch.zeros((waveform.shape[0], pad_length), dtype=waveform.dtype, device=waveform.device)
    padded_waveform = torch.cat([waveform, padding], dim=1)
    return padded_waveform


@dataclass
class ChunkInfo:
    audio_id: str
    frame_offset: int  # In target sample rate
    num_frames: int  # In target sample rate


class ChunkedAudioDataset(Dataset):
    """
    Dataset that loads audio from CSV with optional chunking.

    Args:
        csv_path: Path to the CSV file with columns: audio_id, path, length, sample_rate
        audio_root: Root directory for audio files (prepended to paths in CSV)
        chunk_size: Size of each chunk in frames (None = no chunking)
        hop_size: Hop size between chunks in frames (None = use chunk_size)
        mono: Convert to mono if True
        normalize: Normalize audio if True
        target_sample_rate: Resample to this sample rate if provided
    """

    def __init__(
        self,
        csv_path: str,
        audio_root: str,
        chunk_size: int | None = None,
        hop_size: int | None = None,
        mono: bool = True,
        normalize: bool = True,
        target_sample_rate: int | None = None,
    ):
        self.csv_path = csv_path
        self.audio_root = audio_root
        self.chunk_size = chunk_size
        self.hop_size = hop_size if hop_size is not None else chunk_size
        self.mono = mono
        self.normalize = normalize
        self.target_sample_rate = target_sample_rate

        # Load CSV and compute chunks
        self.file_entries = self._load_csv()
        self.chunks = self._compute_chunks()

        logger.info(f"Loaded dataset from {csv_path}: {len(self.file_entries)} files, {len(self.chunks)} chunks")

    def _load_csv(self) -> dict[str, dict]:
        """Load audio metadata from CSV."""
        entries = {}
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries[row["audio_id"]] = {
                    "path": row["path"],
                    "length": int(row["length"]),
                    "sample_rate": int(row["sample_rate"]),
                }
        return entries

    def _compute_chunks(self) -> list[ChunkInfo]:
        """Compute all chunks from the file entries."""
        chunks = []
        for audio_id, entry in self.file_entries.items():
            length = entry["length"]
            sample_rate = entry["sample_rate"]

            # Adjust length if resampling to target sample rate
            if self.target_sample_rate is not None and sample_rate != self.target_sample_rate:
                length = int(length * self.target_sample_rate / sample_rate)
                sample_rate = self.target_sample_rate

            if self.chunk_size is None or length <= self.chunk_size:
                # No chunking, or file is shorter than chunk size: use entire file
                chunks.append(ChunkInfo(audio_id=audio_id, frame_offset=0, num_frames=length))
            else:
                # Chunking: compute all chunks with last chunk aligned to end
                frame_offset = 0
                while frame_offset + self.chunk_size <= length:
                    chunks.append(ChunkInfo(audio_id=audio_id, frame_offset=frame_offset, num_frames=self.chunk_size))
                    frame_offset += self.hop_size

                # Add the last chunk aligned to the end
                last_start = length - self.chunk_size
                if last_start > frame_offset - self.hop_size:
                    chunks.append(ChunkInfo(audio_id=audio_id, frame_offset=last_start, num_frames=self.chunk_size))

        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> AudioItem:
        """Load and return a single audio chunk."""
        chunk = self.chunks[idx]
        entry = self.file_entries[chunk.audio_id]
        orig_sample_rate = entry["sample_rate"]
        full_path = Path(self.audio_root) / entry["path"]

        # Calculate start frame and num frames in original sample rate
        if self.target_sample_rate is not None and orig_sample_rate != self.target_sample_rate:
            orig_frame_offset = int(chunk.frame_offset * orig_sample_rate / self.target_sample_rate)
            orig_num_frames = int(chunk.num_frames * orig_sample_rate / self.target_sample_rate)
        else:
            orig_frame_offset = chunk.frame_offset
            orig_num_frames = chunk.num_frames

        waveform, sample_rate = _load_audio_internal(
            full_path, frame_offset=orig_frame_offset, num_frames=orig_num_frames
        )

        waveform, sample_rate = preprocess_audio(
            waveform=waveform,
            sample_rate=sample_rate,
            mono=self.mono,
            normalize=self.normalize,
            target_sample_rate=self.target_sample_rate,
        )

        # Pad if necessary (in case file is shorter than expected)
        if self.chunk_size is not None and waveform.shape[1] < self.chunk_size:
            waveform = pad_audio(waveform, self.chunk_size)

        return AudioItem(
            waveform=waveform,
            audio_id=chunk.audio_id,
            path=full_path,
            sample_rate=sample_rate,
            frame_offset=chunk.frame_offset,
        )
