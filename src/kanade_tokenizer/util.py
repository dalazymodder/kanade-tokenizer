import logging
from typing import Literal

import torch
import torch.nn as nn

# Configure logger
logger = logging.getLogger("kanade_tokenizer")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s"))
logger.addHandler(handler)


def get_logger() -> logging.Logger:
    return logger


def freeze_modules(modules: list[nn.Module] | None):
    for module in modules:
        if module is not None:
            for param in module.parameters():
                param.requires_grad = False


def _load_audio_internal(
    path: str, frame_offset: int | None = None, num_frames: int | None = None
) -> tuple[torch.Tensor, int]:
    # TorchAudio >= 2.9.0 removed decoding and encoding capabilities to TorchCodec.
    # See: https://github.com/pytorch/audio/issues/3902
    # waveform, sample_rate = torchaudio.load(path, frame_offset=frame_offset or 0, num_frames=num_frames or -1)

    import soundfile as sf

    with sf.SoundFile(path) as f:
        if frame_offset is not None:
            f.seek(frame_offset)
        frames = f.read(frames=num_frames or -1, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(frames.T)
        sample_rate = f.samplerate
    return waveform, sample_rate


def load_audio(audio_path: str, sample_rate: int = 24000) -> torch.Tensor:
    import torchaudio

    """Load and preprocess audio file."""
    waveform, sr = _load_audio_internal(audio_path)

    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)

    # Resample if necessary
    if sr != sample_rate:
        resampler = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resampler(waveform)

    # Normalize waveform
    max_val = torch.max(torch.abs(waveform)) + 1e-8
    waveform = waveform / max_val  # Normalize to [-1, 1]

    return waveform.squeeze(0)  # Remove channel dimension


def load_vocoder(name: Literal["vocos", "hift"] = "vocos") -> torch.nn.Module:
    if name == "vocos":
        from vocos import Vocos

        model = Vocos.from_pretrained("charactr/vocos-mel-24khz")
        model = model.eval()
        return model
    elif name == "hift":
        from huggingface_hub import hf_hub_download
        from .module.hift import HiFTGenerator

        # Download hte HiFT model from FunAudioLLM/CosyVoice2-0.5B
        model_path = hf_hub_download(repo_id="FunAudioLLM/CosyVoice2-0.5B", filename="hift.pt")
        model = HiFTGenerator()
        model.load_weights(model_path)
        model = model.eval()
        return model
    else:
        raise ValueError(f"Unsupported vocoder name: {name}")


def vocode(vocoder, mel_spectrogram: torch.Tensor) -> torch.Tensor:
    """Convert mel spectrogram to waveform using Vocos vocoder.
    Args:
        vocoder: Pretrained vocoder model.
        mel_spectrogram (torch.Tensor): Input mel spectrogram tensor (..., n_mels, frame).
    Returns:
        torch.Tensor: Generated audio waveform tensor (..., samples).
    """
    mel_spectrogram = mel_spectrogram.to(torch.float32)  # Ensure mel spectrogram is in float32

    vocoder_class_name = vocoder.__class__.__name__
    if "Vocos" in vocoder_class_name:
        generated_waveform = vocoder.decode(mel_spectrogram)
    elif "HiFT" in vocoder_class_name:
        generated_waveform = vocoder.inference(mel_spectrogram)
    else:
        raise ValueError(f"Unsupported vocoder class: {vocoder_class_name}")

    return generated_waveform
