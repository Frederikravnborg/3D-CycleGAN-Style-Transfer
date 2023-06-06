import torch
from torchvision import transforms
from datetime import datetime
currentDateAndTime = datetime.now()
currentTime = currentDateAndTime.strftime("%m.%d.%H.%M.%S")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train_test_10"
VAL_DIR = "data/val"
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 0
NUM_EPOCHS = 300
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_M = f"saved_models/genM_{currentTime}.pth.tar"
CHECKPOINT_GEN_F = f"saved_models/genF_{currentTime}.pth.tar"
CHECKPOINT_DISC_M = f"saved_models/discM_{currentTime}.pth.tar"
CHECKPOINT_DISC_F = f"saved_models/discF_{currentTime}.pth.tar"
N_POINTS = 2048
MAX_DISTANCE = torch.tensor(1.0428)
SAVE_OBJ = True
SAVE_RATE = 100 # Save every SAVE_RATE batches
FOLD_SHAPE = 'plane'
PLANE_SIZE = 1
DISC_WIDTH_REDUCER = 5 #factor must be a power of 2

transform = transforms.Lambda(lambda x: x / MAX_DISTANCE)
