import os
import sys
import torch
import numpy as np
import datetime
import logging
import config
from pathlib import Path
from tqdm import tqdm
from load_data import ObjDataset
from torch.utils.data import DataLoader
from pointnet_model_cls import get_model, get_loss
import logging
import wandb
logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# initialize wandb
wandb.init(
    # set the wandb project where this run will be logged
    project="Fagprojekt",
    
    # track hyperparameters and run metadata
    config={
    "POINT_BATCH_SIZE": config.POINT_BATCH_SIZE,
    "POINT_NUM_EPOCHS": config.POINT_NUM_EPOCHS,
    "POINT_LR": config.POINT_LR
    },
    mode = "online" if config.USE_WANDB else "disabled"
)



# define relu function to be used as activation function
def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def train(loader, classifier, criterion, optimizer, scheduler):
    mean_correct = []
    classifier = classifier.train()

    scheduler.step()

    for batch_id, (female, male) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        optimizer.zero_grad()

        ### FEMALE ###
        female = female.data.numpy()
        female = torch.Tensor(female)
        female = female.transpose(2, 1)
        targetF = torch.zeros(len(female))

        female, targetF = female.to(device).float(), targetF.to(device).float()

        ### MALE ###
        male = male.data.numpy()
        male = torch.Tensor(male)
        male = male.transpose(2, 1)
        targetM = torch.ones(len(male))
        
        male, targetM = male.to(device).float(), targetM.to(device).float()

        predF, trans_featF = classifier(female)
        predM, trans_featM = classifier(male)
        pred = torch.cat((predF, predM))
        target = torch.cat((targetF, targetM)).unsqueeze(0)
        trans_feat = torch.cat((trans_featF, trans_featM))
        # target = torch.from_numpy(np.array([[1-a, a] for a in target])).transpose(1,0)

        loss = criterion(pred.transpose(1,0).squeeze(), target.squeeze().long(), trans_feat)
        loss.backward()

        pred_choice = pred.data.max(1)[1]

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(female.size()[0] + male.size()[0]))
        
        optimizer.step()

    acc = np.mean(mean_correct)
    return acc

def test(model, loader, num_class=2):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()
    
    for batch_id, (female, male) in tqdm(enumerate(loader), total=len(loader)):
        ### FEMALE ###
        targetF = torch.zeros(len(female))
        female, targetF = female.to(device), targetF.to(device)
        female = female.transpose(2, 1)
        predF, _ = classifier(female)

        ### MALE ###
        targetM = torch.ones(len(male))
        male, targetM = male.to(device), targetM.to(device)
        male = male.transpose(2, 1)
        predM, _ = classifier(male)

        pred = torch.cat((predF, predM))
        target = torch.cat((targetF, targetM))
        pred_choice = pred.data.max(1)[1]

        femalemale = torch.cat((female, male))

        for cat in np.unique(target.cpu()):
            cat = int(cat)
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(femalemale[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(male.size()[0] + female.size()[0]))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])

    return class_acc


def main():
    def log_string(str):
        logger.info(str)
        print(str)


    # create directory for saving logs and models
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log_PointNet_cls/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(timestr)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    # handle logging
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s.txt' % log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')

    # define train dataset
    dataset = ObjDataset(
        root_male=config.TRAIN_DIR + "/male", 
        root_female=config.TRAIN_DIR + "/female",
        transform=None,
        n_points=config.N_POINTS
    )
    # define validation dataset
    val_dataset = ObjDataset(
        root_male=config.VAL_DIR + "/male",
        root_female=config.VAL_DIR + "/female",
        transform=None,
        n_points=config.N_POINTS
    )

    # define dataloader for train and validation dataset
    loader = DataLoader(dataset, batch_size=config.POINT_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.POINT_BATCH_SIZE, shuffle=True, pin_memory=True)
    

    num_class = 2

    classifier = get_model(1).to(device) # 1 for binary classification in stead of 1
    criterion = # torch.nn.MSELoss().to(device)
    criterion = get_loss().to(device)
    ### brug binary cross entropy loss eller get_loss

    classifier.apply(inplace_relu)

    # load pretrained model if there is one
    try:
        # exp_dir is defined as Path('./log_PointNet_cls/')
        checkpoint = torch.load("log_PointNet_cls/saved_model_cls/best_model.pth")
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrained model')
    except:
        log_string('No existing model, starting training from scratch...')

    # define optimizer for PointNet classifier
    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=config.POINT_LR,
        betas=(0.5, 0.999),
        eps=1e-08,
        weight_decay=config.POINT_DECAY_RATE
    )

    # define learning rate scheduler 
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    best_test_acc = 0.0

    # train and test loop
    for epoch in range(config.POINT_NUM_EPOCHS):
        # train one epoch and log training accuracy
        train_acc = train(loader = loader, 
                          classifier = classifier, 
                          criterion = criterion, 
                          optimizer = optimizer, 
                          scheduler = scheduler)
        wandb.log({"train_acc": train_acc})

        # disable gradient computation when testing
        with torch.no_grad():
            # test one epoch and log test accuracy
            test_acc = test(classifier.eval(), val_loader)
            if (test_acc >= best_test_acc):
                best_test_acc = test_acc
                best_epoch = epoch + 1
            wandb.log({"Test_acc": test_acc}, commit=False)
            wandb.log({"Best_acc": best_test_acc}, commit=False)
            log_string(f'Test Accuracy: {test_acc}')
            log_string(f'Best Accuracy: {best_test_acc}')
            
            # save model if test accuracy is the best so far
            if (test_acc >= best_test_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'test_acc': test_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

    wandb.finish()
if __name__ == '__main__':
    main()
