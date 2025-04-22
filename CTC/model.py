import torch
import torch.nn as nn


class CustomLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomLSTMCell, self).__init__()
        self.hidden_dim = hidden_dim

        self.W_i = nn.Linear(input_dim, hidden_dim)
        self.U_i = nn.Linear(hidden_dim, hidden_dim)

        self.W_f = nn.Linear(input_dim, hidden_dim)
        self.U_f = nn.Linear(hidden_dim, hidden_dim)

        self.W_g = nn.Linear(input_dim, hidden_dim)
        self.U_g = nn.Linear(hidden_dim, hidden_dim)

        self.W_o = nn.Linear(input_dim, hidden_dim)
        self.U_o = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x_t, h_prev, c_prev):
        i_t = torch.sigmoid(self.W_i(x_t) + self.U_i(h_prev))
        f_t = torch.sigmoid(self.W_f(x_t) + self.U_f(h_prev))
        g_t = torch.tanh(self.W_g(x_t) + self.U_g(h_prev))
        o_t = torch.sigmoid(self.W_o(x_t) + self.U_o(h_prev))

        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)

        return h_t, c_t


class CustomBidirectionalLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(CustomBidirectionalLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm_forward = CustomLSTMCell(input_dim, hidden_dim)
        self.lstm_reverse = CustomLSTMCell(input_dim, hidden_dim)

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()

        # Initialize hidden and cell states
        h_forward = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c_forward = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        h_reverse = torch.zeros(batch_size, self.hidden_dim).to(x.device)
        c_reverse = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        # Process forward direction
        forward_outputs = []
        for t in range(seq_len):
            h_forward, c_forward = self.lstm_forward(x[:, t, :], h_forward, c_forward)
            forward_outputs.append(h_forward)

        # Process reverse direction
        reverse_outputs = []
        for t in range(seq_len - 1, -1, -1):
            h_reverse, c_reverse = self.lstm_reverse(x[:, t, :], h_reverse, c_reverse)
            reverse_outputs.append(h_reverse)

        # Reverse back to match the original sequence order
        reverse_outputs = reverse_outputs[::-1]

        # Stack along time dimension
        forward_stacked = torch.stack(
            forward_outputs, dim=1
        )  # [batch, seq_len, hidden_dim]
        reverse_stacked = torch.stack(
            reverse_outputs, dim=1
        )  # [batch, seq_len, hidden_dim]

        # Concatenate along feature dimension
        outputs = torch.cat(
            [forward_stacked, reverse_stacked], dim=2
        )  # [batch, seq_len, 2*hidden_dim]

        return outputs


class LSTM_CTC_Model(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTM_CTC_Model, self).__init__()

        self.lstm = CustomBidirectionalLSTM(input_dim, hidden_dim)

        # Output layer - maps from hidden dimension (doubled for bidirectional) to output classes
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        # x shape: [batch_size, seq_len, input_dim]
        lstm_out = self.lstm(x)  # [batch_size, seq_len, hidden_dim*2]

        # Apply fully connected layer to each time step
        batch_size, seq_len, _ = lstm_out.size()
        logits = self.fc(lstm_out)  # [batch_size, seq_len, output_dim]

        return logits


class CTC_Loss(nn.Module):
    def __init__(self):
        super(CTC_Loss, self).__init__()
        self.ctc_loss = nn.CTCLoss(reduction="mean", zero_infinity=True)

    def forward(self, logits, targets, input_lengths, target_lengths):
        """
        logits: [batch_size, time_steps, output_dim] -> Raw output from the model
        targets: [batch_size, target_length] -> Target sequence (e.g., transcriptions)
        input_lengths: [batch_size] -> Lengths of the input sequences
        target_lengths: [batch_size] -> Lengths of the target sequences
        """
        # Reshape logits to required shape [time_steps, batch_size, output_dim]
        log_probs = torch.log_softmax(logits, dim=2).permute(1, 0, 2)

        # Convert lengths to CPU tensors as required by CTCLoss
        input_lengths = torch.tensor(input_lengths, dtype=torch.long)
        target_lengths = torch.tensor(target_lengths, dtype=torch.long)

        # Calculate loss
        loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        return loss
