import torch
import torch.nn as nn
import math
import kenlm

class ClippedReLU(nn.Module):
    def __init__(self, clip=20):
        super().__init__()
        self.clip = clip

    def forward(self, x):
        return torch.clamp(torch.relu(x), min=0, max=self.clip)

class CTC_Loss(nn.Module):
    def __init__(self):
        super(CTC_Loss, self).__init__()
        self.ctc_loss = nn.CTCLoss(reduction="mean", zero_infinity=True)

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # T = input sequence (time steps)
        # N = batch size
        # C = number of classes (including blank)
        
        # log_probs expected shape for nn.CTCLoss (T, N, C)
        if log_probs.ndim == 3:
            log_probs = log_probs.permute(1, 0, 2)  # (N, T, C) -> (T, N, C)
        elif log_probs.ndim == 2:
            log_probs = log_probs.unsqueeze(1)  # (N, C) -> (N, 1, C)
            log_probs = log_probs.permute(1, 0, 2)  # (N, 1, C) -> (1, N, C)
            input_lengths = [input_lengths] if not isinstance(input_lengths, (list, tuple)) else input_lengths
            target_lengths = [target_lengths] if not isinstance(target_lengths, (list, tuple)) else target_lengths
        else:
            raise ValueError("log_probs must have 2 or 3 dimensions")

        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)

        T = log_probs.size(0)
        input_lengths = torch.clamp(input_lengths, max=T)

        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        return loss


class Hypothesis:
    def __init__(self, prefix, acoustic_score, lm_score, lm_state):
        self.prefix = prefix
        self.acoustic_score = acoustic_score
        self.lm_score = lm_score
        self.lm_state = lm_state

    def combined_score(self, alpha, beta):
        return self.acoustic_score + alpha * self.lm_score + beta * len(self.prefix)


def beam_search_decoded_with_lm(log_probs, label_to_char, lm_model=None, lm_vocab=None, beam_width=10, alpha=0.5, beta=0.0):
    batch_size, seq_len, num_classes = log_probs.size()
    decoded_batch = []
    blank_idx = 0

    for b in range(batch_size):
        initial_lm_state = None
        if lm_model:
            initial_lm_state = kenlm.State()
            lm_model.BeginSentenceWrite(initial_lm_state)  # Prepare LM to score the hypothesis

        beam = [Hypothesis(prefix="", acoustic_score=0.0, lm_score=0.0, lm_state=initial_lm_state)]

        for t in range(seq_len):
            current_log_probs = log_probs[b, t, :]
            new_beam = []

            for hyp in beam:
                # Consider blank token
                blank_prob = current_log_probs[blank_idx].item()
                blank_hyp = Hypothesis(
                    prefix=hyp.prefix,
                    acoustic_score=hyp.acoustic_score + blank_prob,
                    lm_score=hyp.lm_score,
                    lm_state=hyp.lm_state
                )
                new_beam.append(blank_hyp)

                # Consider character token for each letter
                for char_index in range(1, num_classes):
                    char = label_to_char[char_index]
                    char_prob = current_log_probs[char_index].item()
                    new_prefix = hyp.prefix + char
                    lm_score_char = 0.0
                    current_lm_state = hyp.lm_state
                    new_lm_state = None

                    if lm_model and current_lm_state is not None:
                        try:
                            new_lm_state = kenlm.State()
                            lm_score_char = lm_model.BaseScore(current_lm_state, char, new_lm_state)
                            lm_score_char = lm_score_char * math.log(10)
                        except Exception:
                            lm_score_char = -float('inf')
                            new_lm_state = None

                    new_hyp = Hypothesis(
                        prefix=new_prefix,
                        acoustic_score=hyp.acoustic_score + char_prob,
                        lm_score=hyp.lm_score + lm_score_char,
                        lm_state=new_lm_state
                    )
                    new_beam.append(new_hyp)

            scored_beam = [(h, h.combined_score(alpha, beta)) for h in new_beam]
            sorted_beam = sorted(scored_beam, key=lambda x: x[1], reverse=True)  # scored_beam[i] = (Hypothesis_i, combined_score_i)
            beam = [h[0] for h in sorted_beam[:beam_width]]

        if beam:
            final_scored_beam = [(h, h.combined_score(alpha, beta)) for h in beam]
            best_hypothesis = sorted(final_scored_beam, key=lambda x: x[1], reverse=True)[0][0]
            decoded_batch.append(best_hypothesis.prefix)
        else:
            decoded_batch.append("")

    return decoded_batch



def read_digit_sequences(filename="digit_sequences.txt"):
    sequences = []
    try:
        with open(filename, "r") as f:
            for line in f:
                sequences.append(line.strip())
    except FileNotFoundError:
        print(f"Error: File not found: {filename}")
        return []
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return []
    return sequences
