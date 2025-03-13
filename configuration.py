###  this file contain some configuration  
import torch

DATA_FOLDER = "./"
KEEP_DIFFICULT = True 
CHECKPOINTS = None # model path to checkpoints 
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE =8 
ITERS = 120000
WORKERS = 4 
PRINT_FRAQ = 200 # PRINTING FRQUENCY
LR = 1e-3
DECAY_LR = [80000 , 10000] # DACY LEARNING RATE IN THIS EPOCHS 
MOUMENTUM = 0.9 
DACAY_LR_COEFF = 0.1
WEIGHT_DACAY = 5E-4 
GRAD_CLIP = None 

N_CLASSES = 10
