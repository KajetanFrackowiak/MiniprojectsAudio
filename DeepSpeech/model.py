import torch
import torch.nn as nn
import torch.nn.functional as F
from tools import ClippedReLU


class CustomRNNCell(nn.Module):
    def __init__(self, input_dim, hidden_size):
        super(CustomRNNCell, self).__init__()
        self.W_ih = nn.Linear(input_dim, hidden_size)
        self.W_hh = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        return torch.tanh(self.W_ih(x) + self.W_hh(h))


class CustomRNN(nn.Module):
    def __init__(self, input_dim, hidden_size, bidirectional=False):
        super(CustomRNN, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.fwd_cell = CustomRNNCell(input_dim, hidden_size)
        if bidirectional:
            self.bwd_cell = CustomRNNCell(input_dim, hidden_size)

    def forward(self, x, h0=None):
        batch_size, seq_len, _ = x.size()  # (B, T, D)
        device = x.device

        if h0 is None:
            h0 = torch.zeros(batch_size, self.hidden_size, device=device)

        # Forward direction
        fwd_outputs = []
        h_fwd = h0
        for t in range(seq_len):
            # x (B, T, D), x[:, t, :] (B, D)
            h_fwd = self.fwd_cell(x[:, t, :], h_fwd)  # (B, D)
            fwd_outputs.append(h_fwd.unsqueeze(1))  # (B, 1, H)
        fwd_outputs = torch.cat(fwd_outputs, dim=1)  #  list([B, 1, H], [B, 1, H]...) length T -> (B, T,   H)

        if self.bidirectional:
            bwd_outputs = []
            h_bwd = h0
            for t in reversed(range(seq_len)):
                h_bwd = self.bwd_cell(x[:, t, :], h_bwd)  # (B, D)
                bwd_outputs.insert(0, h_bwd.unsqueeze(1))  # (B, 1, H)
            bwd_outputs = torch.cat(bwd_outputs, dim=1)  # (B, T, H)

            return torch.cat([fwd_outputs, bwd_outputs], dim=2)  # (B, T, 2H)
        else:
            return fwd_outputs  # (B, T, H)


class DeepSpeech(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_rnn_layers=3, context=9, num_initial_layers=3):
        super(DeepSpeech, self).__init__()
        self.context = context
        self.num_initial_layers = num_initial_layers
        self.hidden_dim = hidden_dim

        self.initial_layers = nn.ModuleList()
        input_dim *= context
        for _ in range(num_initial_layers):
            self.initial_layers.append(nn.Linear(input_dim, hidden_dim))
            input_dim = hidden_dim

        self.relu = ClippedReLU(clip=20)

        self.rnns = nn.ModuleList([
            CustomRNN(hidden_dim if i == 0 else hidden_dim * 2, hidden_dim, bidirectional=True)
            for i in range(num_rnn_layers)
        ])

        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, x, input_lengths):
        # x: (B, T, D)
        batch_size, seq_len, feat_dim = x.size()

        # Context windowing
        x = x.transpose(1, 2)  # (B, D, T)
        x = F.pad(x, (self.context // 2, self.context - 1 - self.context // 2), mode="replicate")
        x = x.unfold(dimension=2, size=self.context, step=1)  # (B, D, T, context)
        x = x.permute(0, 2, 1, 3).reshape(batch_size, seq_len, -1)  # (B, T, D * context)

        # Initial non-recurrent layers
        out = x
        
        for layer in self.initial_layers:
            out = self.relu(layer(out))

        # RNN layers
        for rnn in self.rnns:
            out = rnn(out)

        logits = self.fc(out)  # (B, T, output_dim)
        log_probs = self.log_softmax(logits)
        return log_probs  # used for CTC loss
