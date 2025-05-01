import mlflow
import kenlm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from data_preprocessing import load_and_preprocess_data, collate_fn
from config import (
    BATCH_SIZE,
    DEVICE,
    INPUT_DIM,
    HIDDEN_DIM,
    OUTPUT_DIM,
    NUM_RNN_LAYERS,
    NUM_INITIAL_LAYERS,
    CONTEXT_WINDOW,
    LEARNING_RATE,
    EPOCHS,
    ALPHA,
    BETA
)
from model import DeepSpeech
from tools import CTC_Loss, read_digit_sequences
from train import train, evaluate


def main():
    mlflow.set_experiment("DeepSeek")
    print(f"Device: {DEVICE}")
    

    dataset = load_and_preprocess_data()
    train_loader = DataLoader(
        dataset["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn
    )
    test_loader = DataLoader(
        dataset["test"], batch_size=BATCH_SIZE, collate_fn=collate_fn
    )
    model = DeepSpeech(
        INPUT_DIM,
        HIDDEN_DIM,
        OUTPUT_DIM,
        NUM_RNN_LAYERS,
        CONTEXT_WINDOW,
        NUM_INITIAL_LAYERS,
    ).to(DEVICE)
    ctc_loss = CTC_Loss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    lm_model_path = "digit_language_model.bin"
    lm_model = kenlm.Model(lm_model_path)
    lm_vocab_path = "digit_sequences.txt"
    lm_vocab = read_digit_sequences(lm_vocab_path)
    with mlflow.start_run():
        mlflow.log_param("batch_size", BATCH_SIZE)
        mlflow.log_param("epochs", EPOCHS)
        mlflow.log_param("learning_rate", LEARNING_RATE)
        mlflow.log_param("alpha", ALPHA)
        mlflow.log_param("beta", BETA)

        for epoch in range(EPOCHS):
            train(model, train_loader, optimizer, ctc_loss, DEVICE, epoch)
            evaluate(model, test_loader, ctc_loss, DEVICE, lm_model, lm_vocab, epoch)

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimzier.state_dict": optimizer.state_dict(),
                },
                f"checkpoint_{epoch + 1}.pth",
            )

        mlflow.pytorch.log_model(model, "model")
        print("Train finished!")

if __name__ == "__main__":
    main()