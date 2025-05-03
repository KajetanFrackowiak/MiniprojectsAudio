import torch
import soundfile as sf
from data_preprocessing import mu_law_decode
from model import WaveNet
from config import (
    SAMPLE_RATE, QUANTIZATION_CHANNELS, RESIDUAL_CHANNELS, DILATION_CHANNELS, SKIP_CHANNELS, END_CHANNELS, DILATIONS)


def generate_audio(model, length=16000, temperature=1.0, initial_input=None):
    model.eval()

    if initial_input is None:
        current_sample = torch.zeros(1, dtype=torch.long)
    else:
        current_sample = initial_input

    generated_samples = []

    print("Generating audio...")
    for i in range(length):
        if i % 1000 == 0:
            print(f"Generated {i}/{length} samples")

        with torch.no_grad():
            # One-hot encoded input
            x = current_sample.view(1, -1)  # [1, T]
            output = model(x)

            # Probability distribution for the next sample
            output = output[:, :, -1].squeeze()  # [quantization_channels]

            if temperature > 0:
                output = output / temperature

            probabilities = torch.softmax(output, dim=0)
            next_sample = torch.multinomial(probabilities, 1)

        generated_samples.append((next_sample.item()))
        current_sample = torch.cat([current_sample, next_sample])

    # Decode using mu-law
    generated_samples = torch.tensor(generated_samples, dtype=torch.long)
    decoded_samples = mu_law_decode(generated_samples, QUANTIZATION_CHANNELS)

    return decoded_samples.numpy()

def save_audio(audio, filename, sample_rate=SAMPLE_RATE):
    sf.write(filename, audio, sample_rate)

if __name__ == "__main__":
    model = WaveNet(quantization_channels=QUANTIZATION_CHANNELS,
                    residual_channels=RESIDUAL_CHANNELS,
                    dilation_channels=DILATION_CHANNELS,
                    skip_channels=SKIP_CHANNELS,
                    end_channels=END_CHANNELS,
                    dilations=DILATIONS)
    checkpoint = torch.load("wavenet_model.pth", map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    audio = generate_audio(model, length=32000)

    save_audio(audio, "generated_audio.wav")
    print("Audio saved to generated_audio.wav")