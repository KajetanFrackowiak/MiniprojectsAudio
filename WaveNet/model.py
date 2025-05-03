import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        # Calculate the padding needed to ensure casuality.
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        # Remove the future part, we only depend on the past and current inputs
        return self.conv(x)[:, :, :-self.padding]


class WaveNetBlock(nn.Module):
    def __init__(self, residual_channels, dilation_channels, skip_channels, dilation):
        super(WaveNetBlock, self).__init__()
        # Double out_channels to split then on "filter" and "gate" with maintained the length of sequence
        self.causal = CausalConv1d(residual_channels, dilation_channels * 2, kernel_size=2, dilation=dilation)
        self.residual_conv = nn.Conv1d(dilation_channels, residual_channels, kernel_size=1)
        self.skip_conv = nn.Conv1d(dilation_channels, skip_channels, kernel_size=1)

    def forward(self, x):
        causal_output = self.causal(x)  # (batch_size, residual_channels, sequence_length)

        # Gated activation unit
        filter_gate = causal_output.chunk(2, dim=1)
        filter_part, gate_part = filter_gate[0], filter_gate[1]
        gated_output = torch.tanh(filter_part) * torch.sigmoid(gate_part)

        # Residual and skip connections
        residual = self.residual_conv(gated_output)
        skip = self.skip_conv(gated_output)

        return residual + x, skip


class WaveNet(nn.Module):
    def __init__(self, quantization_channels=256, residual_channels=32, dilation_channels=32,
                 skip_channels=256, end_channels=256,
                 dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512] * 2):
        super(WaveNet, self).__init__()

        self.quantization_channels = quantization_channels
        self.dilations = dilations

        # First causal convolution
        self.causal_conv = CausalConv1d(
            quantization_channels, residual_channels, kernel_size=2
        )

        # Stack of dilated convolutions
        self.dilated_blocks = nn.ModuleList([
            WaveNetBlock(residual_channels, dilation_channels, skip_channels, dilation)
            for dilation in dilations
        ])

        # Final output layers
        self.end_conv1 = nn.Conv1d(skip_channels, end_channels, kernel_size=1)
        self.end_conv2 = nn.Conv1d(end_channels, quantization_channels, kernel_size=1)

    def forward(self, x):
        # One-hot encode the input audio
        x = F.one_hot(x, self.quantization_channels).float()
        x = x.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        # Initial causal convolution
        current_input = self.causal_conv(x)

        # Sum of skip connections
        skip_connections = 0

        # Dilated convolution stack
        for layer in self.dilated_blocks:
            current_input, skip = layer(current_input)
            skip_connections += skip

        # Final layers
        output = torch.relu(skip_connections)
        output = torch.relu(self.end_conv1(output))
        output = self.end_conv2(output)

        return output