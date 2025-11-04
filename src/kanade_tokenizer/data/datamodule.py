from dataclasses import dataclass
from pathlib import Path

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from ..util import get_logger
from .dataset import AudioItem, ChunkedAudioDataset, pad_audio

logger = get_logger()


@dataclass
class AudioBatch:
    waveform: torch.Tensor  # [batch, channels, samples]
    audio_ids: list[str]
    paths: list[Path]
    sample_rates: list[int]
    frame_offsets: list[int] | None  # For chunked audio


@dataclass
class AudioDataConfig:
    csv_path: str
    audio_root: str

    # Audio processing
    sample_rate: int | None = 16000
    mono: bool = True
    normalize: bool = True

    # Chunking options
    chunk_size: int | None = None
    chunk_hop_size: int | None = None

    # DataLoader options
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = False
    persistent_workers: bool = False
    shuffle: bool = False
    drop_last: bool = False


def audio_collate_fn(batch: list[AudioItem]) -> AudioBatch:
    waveforms = [item.waveform for item in batch]

    # Pad all waveforms to max length
    max_length = max(wave.shape[1] for wave in waveforms)
    if any(wave.shape[1] != max_length for wave in waveforms):
        waveforms = [pad_audio(wave, max_length) for wave in waveforms]

    return AudioBatch(
        waveform=torch.stack(waveforms),
        audio_ids=[item.audio_id for item in batch],
        paths=[item.path for item in batch],
        sample_rates=[item.sample_rate for item in batch],
        frame_offsets=[item.frame_offset for item in batch],
    )


class AudioDataModule(L.LightningDataModule):
    def __init__(
        self,
        train_config: AudioDataConfig,
        val_config: AudioDataConfig | None = None,
        test_config: AudioDataConfig | None = None,
    ):
        super().__init__()
        self.train_config = train_config
        self.val_config = val_config or train_config
        self.test_config = test_config or self.val_config

        # Set to be initialized in setup()
        self.train_dataset: Dataset | None = None
        self.val_dataset: Dataset | None = None
        self.test_dataset: Dataset | None = None

    def _create_dataset(self, config: AudioDataConfig) -> Dataset:
        return ChunkedAudioDataset(
            csv_path=config.csv_path,
            audio_root=config.audio_root,
            chunk_size=config.chunk_size,
            hop_size=config.chunk_hop_size,
            mono=config.mono,
            normalize=config.normalize,
            target_sample_rate=config.sample_rate,
        )

    def setup(self, stage: str | None = None):
        if stage == "fit" or stage is None:
            self.train_dataset = self._create_dataset(self.train_config)
            self.val_dataset = self._create_dataset(self.val_config)
        elif stage == "validate":
            self.val_dataset = self._create_dataset(self.val_config)
        elif stage == "test" or stage == "predict":
            self.test_dataset = self._create_dataset(self.test_config)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_config.batch_size,
            num_workers=self.train_config.num_workers,
            pin_memory=self.train_config.pin_memory,
            persistent_workers=self.train_config.persistent_workers if self.train_config.num_workers > 0 else False,
            shuffle=self.train_config.shuffle,
            drop_last=self.train_config.drop_last,
            collate_fn=audio_collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_config.batch_size,
            num_workers=self.val_config.num_workers,
            pin_memory=self.val_config.pin_memory,
            persistent_workers=self.val_config.persistent_workers if self.val_config.num_workers > 0 else False,
            shuffle=False,
            drop_last=False,
            collate_fn=audio_collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_config.batch_size,
            num_workers=self.test_config.num_workers,
            pin_memory=self.test_config.pin_memory,
            persistent_workers=self.test_config.persistent_workers if self.test_config.num_workers > 0 else False,
            shuffle=False,
            drop_last=False,
            collate_fn=audio_collate_fn,
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_config.batch_size,
            num_workers=self.test_config.num_workers,
            pin_memory=self.test_config.pin_memory,
            persistent_workers=self.test_config.persistent_workers if self.test_config.num_workers > 0 else False,
            shuffle=False,
            drop_last=False,
            collate_fn=audio_collate_fn,
        )
