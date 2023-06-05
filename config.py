import torch
#import albumentations as A
from torchvision import transforms
#from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 2
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 0.1
NUM_WORKERS = 0
NUM_EPOCHS = 50
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_GEN_M = "genM.pth.tar"
CHECKPOINT_GEN_F = "genF.pth.tar"
CHECKPOINT_CRITIC_M = "criticM.pth.tar"
CHECKPOINT_CRITIC_F = "criticF.pth.tar"
N_POINTS = 2048
MAX_DISTANCE = torch.tensor(1.0428)
SAVE_OBJ = True
SAVE_RATE = 20 # Save every SAVE_RATE batches

transform = transforms.Lambda(lambda x: x / MAX_DISTANCE)