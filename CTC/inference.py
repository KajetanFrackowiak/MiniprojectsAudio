import torch
import torchaudio
import gradio as gr
import numpy as np
from model import LSTM_CTC_Model
from config import input_dim, hidden_dim, output_dim
from torchaudio.transforms import MelSpectrogram

# Load the model
model = LSTM_CTC_Model(input_dim, hidden_dim, output_dim)
checkpoint = torch.load("checkpoint_10.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()


def transcribe(audio):
    # Handle audio input
    if isinstance(audio, tuple):
        audio_path = audio[0]
    else:
        audio_path = audio

    print(f"Loading audio from: {audio_path}")
    waveform, sample_rate = torchaudio.load(audio_path)
    print(f"Original audio shape: {waveform.shape}, sample rate: {sample_rate}")

    # Resample if needed
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
        print(f"Resampled to 16kHz, new shape: {waveform.shape}")

    # Apply mel spectrogram transform
    mel_transform = MelSpectrogram(sample_rate=16000, n_mels=40)
    mel_spec = mel_transform(waveform)
    print(f"Mel spectrogram shape: {mel_spec.shape}")

    # Reshape for model
    mel_spec = mel_spec.squeeze(0).transpose(0, 1).unsqueeze(0)
    print(f"Input to model shape: {mel_spec.shape}")

    with torch.no_grad():
        # Forward pass
        logits = model(mel_spec)
        print(f"Model output shape: {logits.shape}")

        # Get predictions and examine them
        predicted_ids = logits.argmax(dim=-1)[0]
        print(f"Predicted IDs shape: {predicted_ids.shape}")
        print(f"Unique predicted IDs: {torch.unique(predicted_ids).tolist()}")

        # Check how many non-blank predictions
        non_blank_count = (predicted_ids != 0).sum().item()
        print(
            f"Number of non-blank predictions: {non_blank_count} out of {len(predicted_ids)}"
        )

        # Vocabulary
        vocab = [
            "<blank>",
            "A",
            "B",
            "C",
            "D",
            "E",
            "F",
            "G",
            "H",
            "I",
            "J",
            "K",
            "L",
            "M",
            "N",
            "O",
            "P",
            "Q",
            "R",
            "S",
            "T",
            "U",
            "V",
            "W",
            "X",
            "Y",
            "Z",
            " ",
            "'",
        ]

        # Basic CTC decoding
        previous = -1
        transcription = ""
        for i in predicted_ids.cpu().numpy():
            if i != 0 and i != previous:  # Skip blanks and repeated characters
                transcription += vocab[i]
            previous = i

        print(f"Final transcription: '{transcription}'")
        return transcription


# Create Gradio interface
interface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="filepath"),  # Explicitly request filepath
    outputs="text",
    title="Speech Recognition with CTC",
    description="Upload an audio file to transcribe speech to text",
)

if __name__ == "__main__":
    interface.launch()
