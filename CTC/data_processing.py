import aiohttp
from datasets import load_dataset
import torchaudio
import torch
from torchaudio.transforms import MelSpectrogram
from collections import Counter

mel_spectrogram = MelSpectrogram(sample_rate=16000, n_mels=40)

CHAR_VOCAB = {ch: idx + 1 for idx, ch in enumerate("YESNOUP ")}
CHAR_VOCAB["|"] = 0  # Blank token for CTC

LABEL_TO_WORD = {0: "YES", 1: "NO", 2: "UP"}


def load_and_preprocess_data():
    dataset = load_dataset(
        path="speech_commands",
        name="v0.02",
        split={
            "train": "train",  
            "test": "test",
        },
        storage_options={
            "client_kwargs": {"timeout": aiohttp.ClientTimeout(total=3600)}
        },
    )

    # Filter dataset to include only "yes," "no," "up" (labels 0, 1, 2)
    def filter_words(example):
        return example["label"] in [0, 1, 2]

    dataset = dataset.filter(filter_words)

    print(f"Train dataset size after filtering: {len(dataset['train'])}")
    print(f"Test dataset size after filtering: {len(dataset['test'])}")

    for split in ["train", "test"]:
        if len(dataset[split]) == 0:
            raise ValueError(
                f"Filtered {split} dataset is empty. Check filter conditions."
            )

    dataset = dataset.map(preprocess_audio, remove_columns=["file", "label"])

    count_labels(dataset["train"])
    count_labels(dataset["test"])

    return dataset


def preprocess_audio(example):
    audio = example["audio"]["array"]
    sampling_rate = example["audio"]["sampling_rate"]

    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
        audio = resampler(torch.tensor(audio).float())
    else:
        audio = torch.tensor(audio).float()

    mel_spec = mel_spectrogram(audio)

    # Map integer label to uppercase word
    text = LABEL_TO_WORD[example["label"]]

    return {
        "input_values": mel_spec.squeeze(0),  # [n_mels, time]
        "sampling_rate": sampling_rate,
        "text": text,  # Use "text" instead of "label" to avoid schema conflict
    }


def text_to_tensor(text):
    indices = [CHAR_VOCAB[c] for c in text]
    return torch.tensor(indices, dtype=torch.long)

def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = [
        text_to_tensor(item["text"]) for item in batch
    ]  # Use "text" instead of "label"

    # Convert input_values to tensors
    input_values = [
        torch.tensor(i) if not isinstance(i, torch.Tensor) else i for i in input_values
    ]

    input_lengths = [i.shape[1] for i in input_values]
    target_lengths = [len(label) for label in labels]

    max_input_length = max(input_lengths)
    max_target_length = max(target_lengths)

    padded_input_values = []
    for seq in input_values:
        padding_length = max_input_length - seq.shape[1]
        if padding_length > 0:
           padded_seq = torch.nn.functional.pad(seq, (0, padding_length))
        else:
            padded_seq = seq
        padded_input_values.append(padded_seq)

    padded_labels = []
    for seq in labels:
        padding_length = max_target_length - seq.shape[0]
        padded_seq = torch.cat([seq, torch.tensor([CHAR_VOCAB["|"]] * padding_length)])
        padded_labels.append(padded_seq)

    input_values = torch.stack(padded_input_values)  # [batch, n_mels, time]
    labels = torch.stack(padded_labels)  # [batch, max_target_length]

    return input_values, labels, input_lengths, target_lengths


def count_labels(dataset_split):
    label_list = [example["text"] for example in dataset_split]
    label_counts = Counter(label_list)
    
    print("\nLabel counts:")
    for label, count in label_counts.items():
        print(f"{label}: {count}")

