import mlflow
import torch
from data_preprocessing import CHAR_VOCAB
from tqdm import tqdm
from config import EPOCHS, BEAM_WIDTH, ALPHA, BETA
from tools import beam_search_decoded_with_lm

def train(model, train_loader, optimizer, ctc_loss, device, epoch):
    model.train()
    total_loss = 0

    for inputs, targets, input_lengths, target_lengths in tqdm(train_loader, desc=f"Epoch: {epoch+1}/{EPOCHS}"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()

        log_probs = model(inputs, targets)

        loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    mlflow.log_metric("train_loss", avg_loss, step=epoch)
    print(f"Epoch {epoch + 1}, Training Loss: {avg_loss:.4f}")
    
def evaluate(model, test_loader, ctc_loss, device, lm_model, lm_vocab, epoch):
    model.eval()
    total_loss = 0
    correct_beam = 0
    total_examples = 0
    label_to_char = {i: c for c, i in CHAR_VOCAB.items()}

    with torch.no_grad():
        for batch_idx, (inputs, targets, input_lengths, target_lengths) in enumerate(test_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            batch_size = inputs.size(0)
            total_examples += batch_size

            log_probs = model(inputs, targets)
            log_probs_softmax = torch.nn.functional.log_softmax(log_probs, dim=-1)

            loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
            total_loss += loss.item()

            beam_search_decoded_text = beam_search_decoded_with_lm(
                log_probs=log_probs_softmax, 
                label_to_char=label_to_char, 
                lm_model=lm_model,
                lm_vocab=lm_vocab, 
                beam_width=BEAM_WIDTH, 
                alpha=ALPHA,
                beta=BETA
            )

            targets_np = targets.cpu().numpy()
            target_lengths_np = torch.tensor(target_lengths, dtype=torch.long).numpy()
            
            ground_truth_text = []
            for i in range(len(targets_np)):
                valid_indices = targets_np[i][:target_lengths_np[i]]
                text = "".join([label_to_char[idx] for idx in valid_indices])
                ground_truth_text.append(text)

            if batch_idx == 0:
                for i in range(min(5, len(beam_search_decoded_text))):
                    print(f"\nSample {i + 1}")
                    print(f"Beam Decoded: '{beam_search_decoded_text[i]}'")
                    print(f"Ground Truth: '{ground_truth_text[i]}'")

            for beam_decoded, truth in zip(beam_search_decoded_text, ground_truth_text):
                if beam_decoded.lower() == truth.lower():
                    correct_beam += 1

        avg_test_loss = total_loss / len(test_loader)
        beam_accuracy = correct_beam / total_examples if total_examples > 0 else 0.0
        
        mlflow.log_metric("test_loss", avg_test_loss, step=epoch)
        mlflow.log_metric("beam_accuracy", beam_accuracy, step=epoch)

        print(f"Epoch {epoch + 1}, Test Loss: {avg_test_loss:.4f}")
        print(f"Beam Search Decoding Accuracy: {beam_accuracy:.4f} ({correct_beam}/{total_examples})")
