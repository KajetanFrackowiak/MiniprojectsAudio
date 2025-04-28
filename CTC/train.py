import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import LSTM_CTC_Model, CTC_Loss, beam_search_decode, greedy_decode
from data_processing import load_and_preprocess_data, collate_fn, CHAR_VOCAB
from config import input_dim, hidden_dim, output_dim, batch_size, epochs, learning_rate


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

dataset = load_and_preprocess_data()

train_loader = DataLoader(
    dataset["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn
)
test_loader = DataLoader(dataset["test"], batch_size=batch_size, collate_fn=collate_fn)


label_to_char = {i: c for c, i in CHAR_VOCAB.items()}

model = LSTM_CTC_Model(input_dim, hidden_dim, output_dim).to(device)
ctc_loss = CTC_Loss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


def get_ground_truth_from_targets(targets, target_lengths):
    batch_texts = []
    for i in range(len(targets)):
        # Only use valid indices up to the target length
        valid_indices = targets[i][:target_lengths[i]]
        # Convert indices to characters and join them
        text = ''.join([label_to_char[idx.item()] for idx in valid_indices])
        batch_texts.append(text)
    return batch_texts


for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    # Examine first batch for debugging
    if epoch == 0:
        first_batch = next(iter(train_loader))
        inputs, targets, input_lengths, target_lengths = first_batch
        print("\nExamining target sequences in first batch:")
        for i in range(min(5, len(targets))):
            target_seq = targets[i]
            t_len = target_lengths[i]
            print(f"Sample {i+1}:")
            print(f"  Target sequence (indices): {target_seq[:t_len].tolist()}")
            chars = [label_to_char.get(idx.item(), '?') for idx in target_seq[:t_len]]
            word = ''.join(chars)
            print(f"  Target sequence (chars): '{word}'")

    for inputs, targets, input_lengths, target_lengths in tqdm(
        train_loader, desc=f"Epoch: {epoch + 1}/{epochs}"
    ):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        # Rearrange inputs to [batch_size, time_steps, freq_bins]
        inputs = inputs.permute(0, 2, 1).contiguous()  # [batch, time, n_mels]

        outputs = model(inputs)  # [batch_size, seq_len, output_dim]

        loss = ctc_loss(outputs, targets, input_lengths, target_lengths)
        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}")

    model.eval()
    test_loss = 0
    correct_greedy = 0
    correct_beam = 0
    total_examples = 0
    
    with torch.no_grad():
        for inputs, targets, input_lengths, target_lengths in test_loader:
            batch_size = inputs.size(0)
            total_examples += batch_size
            
            inputs, targets = inputs.to(device), targets.to(device)
            inputs = inputs.permute(0, 2, 1).contiguous()
            outputs = model(inputs)  # [batch_size, seq_len, output_dim]
            loss = ctc_loss(outputs, targets, input_lengths, target_lengths)
            test_loss += loss.item()
            
            ground_truth_texts = get_ground_truth_from_targets(targets, target_lengths)
            
            logits = outputs.permute(1, 0, 2).contiguous()  # [time_steps, batch_size, output_dim]
            
            greedy_decoded_texts = greedy_decode(logits, label_to_char)
            for i, (decoded, truth) in enumerate(zip(greedy_decoded_texts, ground_truth_texts)):
                if i < 5 and epoch % 2 == 0:  # Print sample results every other epoch
                    print(f"\nGreedy - Sample {i+1}:")
                    print(f"Decoded: '{decoded}'")
                    print(f"Ground Truth: '{truth}'")
                if decoded == truth:
                    correct_greedy += 1
            
            beam_search_decoded_texts = beam_search_decode(logits, label_to_char)
            for i, (decoded, truth) in enumerate(zip(beam_search_decoded_texts, ground_truth_texts)):
                if i < 5 and epoch % 2 == 0:  # Print sample results every other epoch
                    print(f"\nBeam Search - Sample {i+1}:")
                    print(f"Decoded: '{decoded}'")
                    print(f"Ground Truth: '{truth}'")
                if decoded == truth:
                    correct_beam += 1

        avg_test_loss = test_loss / len(test_loader)
        greedy_accuracy = correct_greedy / total_examples
        beam_accuracy = correct_beam / total_examples
        
        print(f"Epoch {epoch + 1}, Test Loss: {avg_test_loss:.4f}")
        print(f"Greedy Decoding Accuracy: {greedy_accuracy:.4f} ({correct_greedy}/{total_examples})")
        print(f"Beam Search Accuracy: {beam_accuracy:.4f} ({correct_beam}/{total_examples})")

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
        },
        f"checkpoint_{epoch + 1}.pth",
    )

print("Training completed!")