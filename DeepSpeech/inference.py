import torch
import torchaudio
import gradio as gr
import kenlm
from model import DeepSpeech
from config import (
    INPUT_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    NUM_RNN_LAYERS,
    CONTEXT_WINDOW,
    NUM_INITIAL_LAYERS,
    BEAM_WIDTH,
    ALPHA,
    BETA,
    SAMPLE_RATE
)
from data_preprocessing import extract_fbanks, CHAR_VOCAB
from tools import beam_search_decoded_with_lm, read_digit_sequences

model = DeepSpeech(
    INPUT_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    NUM_RNN_LAYERS,
    CONTEXT_WINDOW,
    NUM_INITIAL_LAYERS,
)
checkpoint = torch.load("checkpoint_20.pth", map_location=torch.device("cpu"))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

label_to_char = {i: c for c, i in CHAR_VOCAB.items()}

lm_model_path = "digit_language_model.bin"
lm_model = kenlm.Model(lm_model_path)
lm_vocab_path = "digit_sequences.txt"
lm_vocab = read_digit_sequences(lm_vocab_path)


def transcribe(audio):
    if isinstance(audio, tuple):
        sample_rate, waveform = audio
        waveform = torch.tensor(waveform, dtype=torch.float32)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
             waveform = waveform[0, :].unsqueeze(0)
    else:
        print(f"Loading audio from: {audio}")
        waveform, sample_rate = torchaudio.load(audio)
        if waveform.ndim > 1 and waveform.shape[0] > 1:
             waveform = waveform[0, :].unsqueeze(0)
        elif waveform.ndim == 1:
             waveform = waveform.unsqueeze(0)

    print(f"Original audio shape: {waveform.shape}, sample rate: {sample_rate}")

    if sample_rate != SAMPLE_RATE:
        print(f"Resampling from {sample_rate} Hz to {SAMPLE_RATE} Hz")
        resampler = torchaudio.transforms.Resample(sample_rate, SAMPLE_RATE)
        waveform = resampler(waveform)
        sample_rate = SAMPLE_RATE
        print(f"Resampled audio shape: {waveform.shape}, sample rate: {sample_rate}")

    fbanks = extract_fbanks(waveform, sample_rate)  # Returns shape (1, n_mels, time)
    print(f"Fbanks shape (after extract_fbanks): {fbanks.shape}")

    fbanks_transpose = fbanks.transpose(1, 2)  # Creates shape (1, time, n_mels)
    print(f"Fbanks shape (transposed): {fbanks_transpose.shape}")

    input_lengths = torch.tensor([fbanks_transpose.shape[1]], dtype=torch.long)
    print(f"Input legths: {input_lengths}")

    with torch.no_grad():
        logits = model(fbanks_transpose, input_lengths)
        print(f"Logits shape: {logits.shape}")
        
        beam_trans = beam_search_decoded_with_lm(
            log_probs=logits,
            label_to_char=label_to_char,
            lm_model=lm_model,
            lm_vocab=lm_vocab,
            beam_width=BEAM_WIDTH,
            alpha=ALPHA,
            beta=BETA,
        )

        print(f"Beam search result: {beam_trans}")

    return beam_trans

interface = gr.Interface(
    fn=transcribe,
    inputs=gr.Audio(type="numpy", label="Record or upload audio containing 'yes' or 'no"),
    outputs=gr.Textbox(label="Transcription result"),
    description="Hello,, I'm DeepSpeech"
)

if __name__ == "__main__":
    interface.launch(share=True)