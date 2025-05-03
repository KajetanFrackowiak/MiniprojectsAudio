import torch
import torch.nn as nn
import torch.optim as optim
import mlflow
from tqdm import tqdm

def train(model, data_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}")
    for x, y in progress_bar:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        output = model(x)

        # Reshape for cross entropy loss
        output = output.transpose(1, 2)  # [B, T, quantization_channels] -> [B, quantization_channels, T]
        loss = nn.CrossEntropyLoss()(output.reshape(-1, output.size(-1)), y.reshape(-1))

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": loss.item()})

    avg_loss = total_loss / len(data_loader)
    mlflow.log_metric("train_loss", avg_loss, step=epoch)
    return avg_loss

def evaluate(model, data_loader, device, epoch):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)

            output = model(x)
            output = output.transpose(1, 2)
            loss = nn.CrossEntropyLoss()(output.reshape(-1, output.size(-1)), y.reshape(-1))

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        mlflow.log_metric("val_loss", avg_loss, step=epoch)
        return avg_loss
