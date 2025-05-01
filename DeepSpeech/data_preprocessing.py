import torch
import torch.nn as nn
import torchaudio
from torchaudio.transforms import MelSpectrogram
from datasets import load_dataset
from config import SAMPLE_RATE, INPUT_DIM

CHAR_VOCAB = {
    '|': 0,
    ' ': 1,
    'e': 2,
    'n': 3,
    'o': 4,
    's': 5,
    'y': 6,
}

LABEL_TO_WORD = {
    0: "no",
    1: "yes",
}

WORD_TO_LABEL = {
    "no": 0,
    "yes": 1,
}

def extract_fbanks(waveform, sample_rate):
    n_fft = 400  # 25ms frame at 16kHz
    hop_length = 160  # 10ms hop
    n_mels = INPUT_DIM  # Filter banks

    mel_spectogram = MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        window_fn=torch.hamming_window,
        center=True,
        power=2.0,
        norm="slaney",
        mel_scale="slaney"
    )

    mel_spec = mel_spectogram(waveform)

    # Convert to filterbank energies
    fbanks = torch.log(torch.clamp(mel_spec, min=1e-5))

    return fbanks

def load_and_preprocess_data():
    dataset = load_dataset(
        path="speech_commands",
        name="v0.02",
        split={
            "train": "train",
            "test": "test",
        }
    )

    def filter_words(example):
        return example["label"] in [0, 1]  # 0: "no", 1: "yes"

    dataset = dataset.filter(filter_words)

    def preprocess_audio(example):
        audio = example["audio"]["array"]
        sampling_rate = example["audio"]["sampling_rate"]

        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(sampling_rate, 16000)
            audio = resampler(torch.tensor(audio).float())
        else:
            audio = torch.tensor(audio).float()

        fbanks = extract_fbanks(audio, 16000)
        fbanks = fbanks.transpose(0, 1)

        text = LABEL_TO_WORD[example["label"]]

        return {
            "input_values": fbanks,  # [time, n_fbanks]
            "sampling_rate": sampling_rate,
            "text": text,
        }

    dataset = dataset.map(preprocess_audio, remove_columns=["file", "label"])

    print(f"Train dataset size after filtering: {len(dataset['train'])}")
    print(f"Test dataset size after filtering: {len(dataset['test'])}")

    for split in ["train", "test"]:
        if len(dataset[split]) == 0:
            raise ValueError(
                f"Filtered {split} dataset is empty. Check filter conditions."
            )
    return dataset

def text_to_tensor(text):
    indices = [CHAR_VOCAB[c] for c in text.lower() if c in CHAR_VOCAB]
    return torch.tensor(indices, dtype=torch.long)

def collate_fn(batch):
    input_values = [item["input_values"] for item in batch]
    labels = [
        text_to_tensor(item["text"]) for item in batch
    ]
    input_values = [
        torch.tensor(i) if not isinstance(i, torch.Tensor) else i for i in input_values
    ]

    input_lengths = [iv.shape[0] for iv in input_values]
    target_lengths = [len(label) for label in labels]

    padded_input_values = nn.utils.rnn.pad_sequence(
        input_values,
        batch_first=True,
        padding_value=0.0,
    )

    padded_labels = nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=CHAR_VOCAB["|"] if "|" in CHAR_VOCAB else 0
    )

    return padded_input_values, padded_labels, input_lengths, target_lengths

