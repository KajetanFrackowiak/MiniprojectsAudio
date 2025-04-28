import torch
import torchaudio
import gradio as gr
from model import LSTM_CTC_Model
from torchaudio.transforms import MelSpectrogram
from config import input_dim, hidden_dim, output_dim
from data_processing import CHAR_VOCAB
from model import beam_search_decode, greedy_decode, remove_blanks, remove_repetitions

model = LSTM_CTC_Model(input_dim, hidden_dim, output_dim)
checkpoint = torch.load("checkpoint_10.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

label_to_char = {i: c for c, i in CHAR_VOCAB.items()}

def transcribe(audio):
    # Handle audio input
    if isinstance(audio, tuple):
        sample_rate, waveform = audio  # Gradio may return (sample_rate, numpy_array)
        waveform = torch.tensor(waveform).float()
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)  # Add channel dimension
    else:
        # Assume audio is a file path
        print(f"Loading audio from: {audio}")
        waveform, sample_rate = torchaudio.load(audio)

    print(f"Original audio shape: {waveform.shape}, sample rate: {sample_rate}")
    
    # Make sure audio isn't too short
    if waveform.shape[1] < 8000:  # At least 0.5 seconds at 16kHz
        print("Audio too short, padding...")
        padded = torch.zeros(1, 8000)
        padded[0, :waveform.shape[1]] = waveform[0, :] 
        waveform = padded

    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(sample_rate, 16000)
        waveform = resampler(waveform)
        sample_rate = 16000
        print(f"Resampled to 16kHz, new shape: {waveform.shape}")

    # Ensure waveform is single-channel
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # Average channels
        print(f"Converted to mono, new shape: {waveform.shape}")

    # Normalize audio (important for consistent predictions)
    if waveform.abs().max() > 0:
        waveform = waveform / waveform.abs().max()

    mel_transform = MelSpectrogram(sample_rate=16000, n_mels=40)
    mel_spec = mel_transform(waveform)
    print(f"Mel spectrogram shape: {mel_spec.shape}")

    # Reshape for model: [batch_size=1, time_steps, n_mels]
    mel_spec = mel_spec.squeeze(0).transpose(0, 1).unsqueeze(0)  # [1, time, n_mels]
    print(f"Input to model shape: {mel_spec.shape}")

    with torch.no_grad():
        # Forward pass
        logits = model(mel_spec)  # [batch_size=1, seq_len, output_dim]
        print(f"Model output shape: {logits.shape}")

        logits_permuted = logits.permute(0, 1, 2)  # Make sure shape is [batch, time, classes]
        beam_trans = beam_search_decode(logits_permuted, label_to_char)
        greedy_trans = greedy_decode(logits_permuted, label_to_char)
        
        beam_trans = remove_blanks(beam_trans, blank_token_index=-1)
        greedy_trans = remove_blanks(greedy_trans, label_to_char)

        beam_trans = remove_repetitions(beam_trans)
        greedy_trans = remove_repetitions(greedy_trans)

        print(f"Greedy decoding: '{greedy_trans}'")
        print(f"Beam search result: '{beam_trans}'")
        
        return beam_trans

interface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="numpy", label="Record or upload audio containing 'yes', 'no', or 'up'"),
    outputs=gr.Textbox(label="Transcription Result"),
    title="Speech Command Recognition",
    description="Speak one of the words: 'yes', 'no', or 'up'. The model will transcribe what you said.",
)

if __name__ == "__main__":
    interface.launch()