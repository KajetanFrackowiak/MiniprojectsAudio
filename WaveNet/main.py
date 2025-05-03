import mlflow
import torch
import torch.optim as optim
from data_preprocessing import load_data
from model import WaveNet
from train import train, evaluate
from config import (
    QUANTIZATION_CHANNELS,
    DILATIONS,
    RESIDUAL_CHANNELS,
    DILATION_CHANNELS,
    SKIP_CHANNELS,
    END_CHANNELS,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    DEVICE,
    CHECKPOINT_INTERVAL
)

def main():
    mlflow.set_experiment("WaveNet")

    with mlflow.start_run():
        mlflow.log_params({
            "quantization_channels": QUANTIZATION_CHANNELS,
            "residual_channels": RESIDUAL_CHANNELS,
            "dilation_channels": DILATION_CHANNELS,
            "skip_channels": SKIP_CHANNELS,
            "end_channels": END_CHANNELS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS
        })

        model = WaveNet(
            quantization_channels=QUANTIZATION_CHANNELS,
            residual_channels=RESIDUAL_CHANNELS,
            dilation_channels=DILATION_CHANNELS,
            skip_channels=SKIP_CHANNELS,
            end_channels=END_CHANNELS,
            dilations=DILATIONS
        ).to(DEVICE)

        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

        train_loader, val_loader = load_data(BATCH_SIZE)

        print(f"Training on {DEVICE}...")

        for epoch in range(EPOCHS):
            train_loss = train(model, train_loader, optimizer, DEVICE, epoch)
            val_loss = evaluate(model, val_loader, DEVICE, epoch)

            print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            if (epoch + 1) % CHECKPOINT_INTERVAL == 0:
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss
                }
                torch.save(checkpoint, f"wavenet_checkpoint_{epoch+1}.pth")

            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, "wavenet_model.pth")

            mlflow.pytorch.log_model(model, "model")

        print("Training complete!")

if __name__ == "__main__":
    main()