import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import LSTM_CTC_Model, CTC_Loss
from data_processing import load_and_preprocess_data, collate_fn
from config import input_dim, hidden_dim, output_dim, batch_size, epochs, learning_rate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = load_and_preprocess_data()

train_loader = DataLoader(
    dataset["train"], batch_size=batch_size, collate_fn=collate_fn
)
test_lodaer = DataLoader(dataset["test"], batch_size=batch_size, collate_fn=collate_fn)

model = LSTM_CTC_Model(input_dim, hidden_dim, output_dim).to(device)
ctc_loss = CTC_Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for inputs, targets, input_lengths, target_lengths in tqdm(
        train_loader, desc=f"Epoch: {epoch + 1}/{epochs}"
    ):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        inputs = inputs.permute(
            0, 2, 1
        ).contiguous()  # rearenge to (batch_size, time_steps, freq_bins)
        # print("Input shape after permutation: ", inputs.shape)
        outputs = model(inputs)

        loss = ctc_loss(outputs, targets, input_lengths, target_lengths)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

torch.save(
    {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": total_loss / len(train_loader),
    },
    f"checkpoint_{epoch + 1}.pth",
)
