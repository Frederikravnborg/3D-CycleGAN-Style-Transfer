import torch
from torchvision import transforms
from datetime import datetime
currentDateAndTime = datetime.now()
currentTime = currentDateAndTime.strftime("%m.%d.%H.%M.%S")

# Training Loop:
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train_test_100"
VAL_DIR = "data/val"
N_POINTS = 2048
BATCH_SIZE = 2
LEARNING_RATE = 1e-4
NUM_WORKERS = 0
MAX_DISTANCE = torch.tensor(1.0428)
transform = transforms.Lambda(lambda x: x / MAX_DISTANCE)

LOAD_MODEL = False
SAVE_MODEL = True

CHECKPOINT_GEN_M = f"saved_models/genM_{currentTime}.pth.tar"
CHECKPOINT_GEN_F = f"saved_models/genF_{currentTime}.pth.tar"
CHECKPOINT_DISC_M = f"saved_models/discM_{currentTime}.pth.tar"
CHECKPOINT_DISC_F = f"saved_models/discF_{currentTime}.pth.tar"

# FoldingNet:
TRAIN_FOLD = True
FOLD_NUM_EPOCH = 6
FOLD_SAVE_OBJ = True
LOAD_FOLD_MODEL = False
CHECKPOINT_FOLD_M = f"pre_saved_models/genM_{currentTime}.pth.tar"
CHECKPOINT_FOLD_F = f"pre_saved_models/genF_{currentTime}.pth.tar"
timestamp = "06.06.19.04.34_50"
SAVEDMODEL_GEN_M = f"pre_saved_models/genM_{timestamp}.pth.tar"
SAVEDMODEL_GEN_F = f"pre_saved_models/genF_{timestamp}.pth.tar"


LAMBDA_CYCLE = 10
PLANE_SIZE = 1
DISC_WIDTH_REDUCER = 5 #factor must be a power of 2

# GAN:
TRAIN_GAN = False
GAN_NUM_EPOCHS = 4
SAVE_OBJ = True
SAVE_RATE = 100 # Save every SAVE_RATE batches
FOLD_SHAPE = 'sphere'
PLANE_SIZE = 1
