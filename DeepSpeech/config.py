import torch

INPUT_DIM = 26  # Filter bank energy feature size
HIDDEN_DIM = 256
OUTPUT_DIM = 7
NUM_RNN_LAYERS = 3
NUM_INITIAL_LAYERS = 3
CONTEXT_WINDOW = 9
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 1e-3
BEAM_WIDTH = 1
ALPHA = 0.5
BETA = 0.0
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available else "cpu")
