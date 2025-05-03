import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader
import os
from datasets import load_dataset
from config import SAMPLE_RATE, SEGMENT_LENGTH, QUANTIZATION_CHANNELS

def mu_law_encode(x, quantization_channels):
    mu = quantization_channels - 1
    safe_x = torch.clamp(torch.abs(x), 0, 1)
    compressed = torch.sign(x) * torch.log1p(mu * safe_x) / torch.log1p(torch.tensor(mu, dtype=torch.float))
    signal = ((compressed + 1) / 2* mu + 0.5).to(torch.int64)
    return signal

def mu_law_decode(y, quantization_channels):
    mu = quantization_channels - 1
    y = y.float()
    y = 2 * (y / mu) - 1
    x = torch.sign(y) * (torch.exp(torch.abs(y) * torch.log1p(torch.tensor(mu, dtype=torch.float))) - 1) / mu
    return x


class AudioDataset(Dataset):
    def __init__(self, audio_dir, sample_rate=16000, segment_length=16000, quantization_channels=256):
        self.audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir)
                            if f.endswith('.wav') or f.endswith('.mp3')]
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.quantization_channels = quantization_channels

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        waveform, sr = torchaudio.load(audio_path)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # Convert to mono if needed
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        waveform = waveform[0]
        waveform = waveform / torch.max(torch.abs(waveform))

        # Randomly select segment
        if waveform.shape[0] > self.segment_length:
            start = np.random.randint(0, waveform.shape[0] - self.segment_length)
            waveform = waveform[start:start + self.segment_length]
        else:
            # Pad if too short
            pad_length = self.segment_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        quantized = mu_law_encode(waveform, self.quantization_channels)

        x = quantized[:-1]
        y = quantized[1:]

        return x, y

class HuggingFaceAudioDataset(Dataset):
    def __init__(self, hf_dataset, sample_rate=16000, segment_length=16000, quantization_channels=256):
        self.dataset = hf_dataset
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.quantization_channels = quantization_channels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        audio_array = np.array(item["audio"]["array"])
        waveform = torch.tensor(audio_array).float()
        sr = item["audio"]["sampling_rate"]

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        if torch.max(torch.abs(waveform)) > 0:
            waveform = waveform / torch.max(torch.abs(waveform))

        if waveform.shape[0] > self.segment_length:
            start = np.random.randint(0, waveform.shape[0] - self.segment_length)
            waveform = waveform[start:start + self.segment_length]
        else:
            pad_length = self.segment_length - waveform.shape[0]
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))

        quantized = mu_law_encode(waveform, self.quantization_channels)

        # Shifted by one for predicting next sample based on previous
        x = quantized[:-1]
        y = quantized[1:]

        return x, y

def load_data(batch_size=16, sample_rate=SAMPLE_RATE,
              segment_length=SEGMENT_LENGTH, quantization_channels=QUANTIZATION_CHANNELS):
    print("Loading dataset...")

    dataset = load_dataset("keithito/lj_speech", split="train[:2000]")

    dataset = dataset.train_test_split(test_size=0.1)

    train_dataset = HuggingFaceAudioDataset(
        dataset["train"],
        sample_rate=sample_rate,
        segment_length=segment_length,  # Using 4000 from config
        quantization_channels=quantization_channels
    )

    val_dataset = HuggingFaceAudioDataset(
        dataset["test"],
        sample_rate=sample_rate,
        segment_length=segment_length,
        quantization_channels=quantization_channels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    return train_loader, val_loader