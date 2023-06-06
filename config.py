import torch
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
<<<<<<< HEAD
NUM_WORKERS = 14
NUM_EPOCHS = 600
LOAD_MODEL = False
=======
NUM_WORKERS = 0
NUM_EPOCHS = 1
LOAD_MODEL = True
>>>>>>> 5f2d5dff04f417725647e7a119e8d12e6e532c7b
SAVE_MODEL = True
CHECKPOINT_GEN_M = "saved_models/genM.pth.tar"
CHECKPOINT_GEN_F = "saved_models/genF.pth.tar"
CHECKPOINT_CRITIC_M = "saved_models/criticM.pth.tar"
CHECKPOINT_CRITIC_F = "saved_models/criticF.pth.tar"
N_POINTS = 2048
MAX_DISTANCE = torch.tensor(1.0428)
SAVE_OBJ = True
SAVE_RATE = 100 # Save every SAVE_RATE batches

transform = transforms.Lambda(lambda x: x / MAX_DISTANCE)
