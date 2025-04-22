import aiohttp
from datasets import load_dataset
import torchaudio
import torch
from torchaudio.transforms import MelSpectrogram

mel_spectogram = MelSpectrogram(sample_rate=16000, n_mels=40)


def load_and_preprocess_data():
    dataset = load_dataset(
        path="librispeech_asr",
        name="clean",
        split={"train": "train.100[:1%]", "test": "test[:1%]"},
        storage_options={
            "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
        },
    )
    dataset = dataset.map(preprocess_audio, remove_columns=["file", "text"])
    return dataset


def preprocess_audio(example):
    audio = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]

    # Convert to Mel spectrogram
    waveform = torch.tensor(audio).float()
    mel_spec = mel_spectogram(waveform)

    return {
        "input_values": mel_spec.squeeze(0),
        "sampling_rate": sampling_rate,
        "label": example["text"],
    }


CHAR_VOCAB = {ch: idx + 1 for idx, ch in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ '")}
CHAR_VOCAB["|"] = 0  # For blank token in CTC


def text_to_tensor(text):
    return torch.tensor([CHAR_VOCAB[c] for c in text], dtype=torch.long)


def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = [text_to_tensor(item["label"]) for item in batch]

    # Convert to tensors first if they aren't already
    input_values = [
        torch.tensor(i) if not isinstance(i, torch.Tensor) else i for i in input_values
    ]

    # Get the lengths (time dimension is the second dimension for spectrograms)
    input_lengths = [
        i.shape[1] if len(i.shape) > 1 else i.shape[0] for i in input_values
    ]
    target_lengths = [len(l) for l in labels]

    # Determine the max length in the batch (input and target)
    max_input_length = max(input_lengths)
    max_target_length = max(target_lengths)

    padded_input_values = []
    for seq in input_values:
        if len(seq.shape) > 1:  # 2D tensor (mel spectrogram)
            padding_length = max_input_length - seq.shape[1]
            if padding_length > 0:
                # Pad along the time dimension (dim=1)
                padded_seq = torch.nn.functional.pad(seq, (0, padding_length))
            else:
                padded_seq = seq
        else:  # 1D tensor
            padding_length = max_input_length - seq.shape[0]
            if padding_length > 0:
                padded_seq = torch.cat([seq, torch.zeros(padding_length)])
            else:
                padded_seq = seq
        padded_input_values.append(padded_seq)

    padded_labels = []
    for seq in labels:
        padding_length = max_target_length - seq.shape[0]
        padded_seq = torch.cat([seq, torch.tensor([CHAR_VOCAB["|"]] * padding_length)])
        padded_labels.append(padded_seq)

    # Stack properly now that dimensions match
    input_values = torch.stack(padded_input_values)
    labels = torch.stack(padded_labels)

    return input_values, labels, input_lengths, target_lengths
