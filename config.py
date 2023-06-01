import torch
#import albumentations as A
from torchvision import transforms
#from albumentations.pytorch import ToTensorV2

#DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"
VAL_DIR = "data/val"
BATCH_SIZE = 32
LEARNING_RATE = 1e-5
LAMBDA_IDENTITY = 0.0
LAMBDA_CYCLE = 10
NUM_WORKERS = 0
NUM_EPOCHS = 11
LOAD_MODEL = True
SAVE_MODEL = True
CHECKPOINT_GEN_M = "genh.pth.tar"
CHECKPOINT_GEN_F = "genz.pth.tar"
CHECKPOINT_CRITIC_M = "critich.pth.tar"
CHECKPOINT_CRITIC_F = "criticz.pth.tar"
N_POINTS = 1024
MAX_DISTANCE = torch.tensor(1.0428)

transform = transforms.Lambda(lambda x: x / MAX_DISTANCE)