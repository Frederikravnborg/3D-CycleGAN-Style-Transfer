import torch
from datetime import datetime
currentDateAndTime = datetime.now()
currentTime = currentDateAndTime.strftime("%m.%d.%H.%M.%S")

# Training Loop:
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = "data/train"                                                # path to training data from config file place in directory
VAL_DIR = "data/val"                                                    # path to validation data from config file place in directory
N_POINTS = 2048                                                         # number of points to sample from each point cloud                           
BATCH_SIZE = 32                                                         # batch size for training                            
LEARNING_RATE = 1e-4                                                    # learning rate for training            
DISC_LR_FACTOR = 1                                                      # factor to multiply learning rate by for discriminator
NUM_WORKERS = 12                                                        # number of workers for data loader         
MAX_DISTANCE = torch.tensor(1.0428)                                     # max distance for point cloud normalization        
USE_WANDB = True                                                        # use wandb for logging            

# GAN:
TRAIN_GAN = True                                                        # train GAN or not 
GAN_NUM_EPOCHS = 300                                                    # number of epochs to train for
LOAD_MODEL = False                                                      # load model from checkpoint, or not
SAVE_MODEL = False                                                      # save model after training, or not   
SAVE_OBJ = True                                                         # save obj files after training, or not              
SAVE_RATE = 1                                                           # Save pair of pcds every SAVE_RATE epochs
DISC_WIDTH_REDUCER = 1                                                  # factor must be a power of 2
CHECKPOINT_GEN_M = f"saved_models/genM_{currentTime}.pth.tar"           # checkpoint file for generator M
CHECKPOINT_GEN_F = f"saved_models/genF_{currentTime}.pth.tar"           # checkpoint file for generator F
CHECKPOINT_DISC_M = f"saved_models/discM_{currentTime}.pth.tar"         # checkpoint file for discriminator M
CHECKPOINT_DISC_F = f"saved_models/discF_{currentTime}.pth.tar"         # checkpoint file for discriminator F

# FoldingNet:
TRAIN_FOLD = False                                                      # train FoldingNet or not             
FOLD_NUM_EPOCH = 0                                                      # number of epochs to train for
FOLD_SAVE_OBJ = True                                                    # save obj files after training foldingnet, or not             
LOAD_FOLD_MODEL = False                                                 # load model from checkpoint, or not 
CHECKPOINT_FOLD_M = f"pre_saved_models/genM_{currentTime}.pth.tar"      # checkpoint file for generator M
CHECKPOINT_FOLD_F = f"pre_saved_models/genF_{currentTime}.pth.tar"      # checkpoint file for generator F
timestamp = "06.06.19.04.34_50"                                         # timestamp for pre-trained models (for loading models)       
SAVEDMODEL_GEN_M = f"pre_saved_models/genM_{timestamp}.pth.tar"         # file name for pre-trained generator M
SAVEDMODEL_GEN_F = f"pre_saved_models/genF_{timestamp}.pth.tar"         # file name for pre-trained generator F
FOLD_SHAPE = 'sphere'                                                   # input shape for foldingnet                 
LAMBDA_CYCLE = 1120                                                     # lambda for cycle loss              
PLANE_SIZE = 1                                                          # size of plane for foldingnet                    

