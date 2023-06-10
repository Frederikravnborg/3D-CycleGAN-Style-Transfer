import os
import sys
import torch
import torch.nn as nn
import numpy as np
import datetime
import logging
import config
from pathlib import Path
from tqdm import tqdm
from torchvision import transforms
from load_data import ObjDataset
from torch.utils.data import DataLoader
from pointnet_model_cls import get_model, get_loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def train(epoch, loader, classifier, criterion, optimizer, scheduler):
    mean_correct = []
    classifier = classifier.train()

    scheduler.step()

    for batch_id, (female, male) in tqdm(enumerate(loader), total=len(loader), smoothing=0.9):
        ### FEMALE ###
        female = female.data.numpy()
        female = torch.Tensor(female)
        female = female.transpose(2, 1)
        targetF = torch.zeros(len(female))
        if config.DEVICE == "cuda":
            female, targetF = female.cuda(), targetF.cuda()

        ### MALE ###
        male = male.data.numpy()
        male = torch.Tensor(male)
        male = male.transpose(2, 1)
        targetM = torch.ones(len(male))
        if config.DEVICE == 'cuda':
            male, targetM = male.cuda(), targetM.cuda()

        predF, trans_featF = classifier(female)
        predM, trans_featM = classifier(male)
        pred = torch.cat((predF, predM))
        trans_feat = torch.cat((trans_featF, trans_featM))
        target = torch.cat((targetF, targetM))

        loss = criterion(pred, target.long().unsqueeze(0).transpose(0, 1).float())
        pred_choice = pred.data.max(1)[1]

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(female.size()[0] + male.size()[0]))
        
        optimizer.zero_grad()
        loss.backward()
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
        if config.DEVICE == "cuda":
            female, targetF = female.cuda(), targetF.cuda()
        female = female.transpose(2, 1)
        predF, _ = classifier(female)

        ### MALE ###
        targetM = torch.ones(len(male))
        if config.DEVICE == "cuda":
            male, targetM = male.cuda(), targetM.cuda()
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

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = config.DEVICE

    '''CREATE DIR'''
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

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s.txt' % log_dir)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')

    '''DATA LOADING'''
    transform = transforms.Lambda(lambda x: x / config.MAX_DISTANCE)
    # define train dataset
    dataset = ObjDataset(
        root_male=config.TRAIN_DIR + "/male", 
        root_female=config.TRAIN_DIR + "/female",
        transform=transform,
        n_points=config.N_POINTS
    )
    # define validation dataset
    val_dataset = ObjDataset(
        root_male=config.VAL_DIR + "/male",
        root_female=config.VAL_DIR + "/female",
        transform=transform,
        n_points=config.N_POINTS
    )

    # define dataloader for train and validation dataset
    loader = DataLoader(dataset, batch_size=config.POINT_BATCH_SIZE, shuffle=True, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config.POINT_BATCH_SIZE, shuffle=True, pin_memory=True)
    
    '''MODEL LOADING'''
    num_class = 2

    classifier = get_model(num_class)
    criterion = torch.nn.MSELoss()
    classifier.apply(inplace_relu)

    if config.DEVICE == "cuda":
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        # exp_dir is defined as Path('./log_PointNet_cls/')
        checkpoint = torch.load("log_PointNet_cls/saved_model_cls/best_model.pth")
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')

    optimizer = torch.optim.Adam(
        classifier.parameters(),
        lr=config.POINT_LR,
        betas=(0.9, 0.999),
        eps=1e-08,
        weight_decay=config.POINT_DECAY_RATE
    )


    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    best_class_acc = 0.0

    '''TRANING'''
    for epoch in range(config.POINT_NUM_EPOCHS):
        accuracy = train(epoch=epoch,
                         loader=loader, 
                         classifier=classifier, 
                         criterion = criterion, 
                         optimizer = optimizer, 
                         scheduler = scheduler)

        with torch.no_grad():
            class_acc = test(classifier.eval(), val_loader)

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
                best_epoch = epoch + 1
            log_string(f'Test Accuracy: {class_acc}')
            log_string(f'Best Accuracy: {best_class_acc}')

            if (class_acc >= best_class_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)


if __name__ == '__main__':
    main()
