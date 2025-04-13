import torch

# Data settings
DATA_FOLDER = "./"
LABELS_FOLDER = "./"  # Added missing labels folder path
KEEP_DIFFICULT = True

# Model settings
CHECKPOINTS = None  # model path to checkpoints
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
N_CLASSES = 10

# Training settings
BATCH_SIZE = 8
ITERS = 120000
WORKERS = 4
PRINT_FREQ = 200  # PRINTING FREQUENCY (fixed typo)
GRAD_CLIP = None

# Optimizer settings
LR = 1e-3
MOMENTUM = 0.9  # Fixed typo
WEIGHT_DECAY = 5e-4  # Fixed typo

# Learning rate schedule
DECAY_LR_AT = [80000, 100000]  # DECAY LEARNING RATE AT THESE ITERATIONS (fixed typo and renamed for clarity)
DECAY_LR_COEFF = 0.1  # Fixed typo