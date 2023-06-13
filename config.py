import torch
from datetime import datetime
currentDateAndTime = datetime.now()
currentTime = currentDateAndTime.strftime("%m.%d.%H.%M.%S")

# Training Loop:
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
N_POINTS = 2048
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
DISC_LR_FACTOR = 1
NUM_WORKERS = 12
MAX_DISTANCE = torch.tensor(1.0428)
USE_WANDB = True



# GAN:
TRAIN_GAN = False
GAN_NUM_EPOCHS = 0
LOAD_MODEL = True
SAVE_MODEL = True
SAVE_OBJ = True
SAVE_RATE = 1 # Save pair of pcds every SAVE_RATE epochs
DISC_WIDTH_REDUCER = 1 #factor must be a power of 2
CHECKPOINT_GEN_M = f"saved_models/genM_{currentTime}.pth.tar"
CHECKPOINT_GEN_F = f"saved_models/genF_{currentTime}.pth.tar"
CHECKPOINT_DISC_M = f"saved_models/discM_{currentTime}.pth.tar"
CHECKPOINT_DISC_F = f"saved_models/discF_{currentTime}.pth.tar"

# FoldingNet:
TRAIN_FOLD = False
FOLD_NUM_EPOCH = 0
FOLD_SAVE_OBJ = True
LOAD_FOLD_MODEL = False 
CHECKPOINT_FOLD_M = f"pre_saved_models/genM_{currentTime}.pth.tar"
CHECKPOINT_FOLD_F = f"pre_saved_models/genF_{currentTime}.pth.tar"
timestamp = "06.06.19.04.34_50"
SAVEDMODEL_GEN_M = f"pre_saved_models/genM_{timestamp}.pth.tar"
SAVEDMODEL_GEN_F = f"pre_saved_models/genF_{timestamp}.pth.tar"
FOLD_SHAPE = 'sphere'
LAMBDA_CYCLE = 1120
PLANE_SIZE = 1

# PointNet classifier:
POINT_NUM_EPOCHS = 5
POINT_BATCH_SIZE = 2
POINT_LR = 1e-4
POINT_DECAY_RATE = 1e-4

