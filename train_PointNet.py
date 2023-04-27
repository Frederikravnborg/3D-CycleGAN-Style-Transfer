import os
import sys
import torch
import numpy as np

import datetime
import logging
import importlib
import shutil
import argparse

from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from loaddata import all_train, all_test

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true',       default=True, help='use cpu mode')
    parser.add_argument('--gpu', type=str,                      default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int,               default=32, help='batch size in training')
    parser.add_argument('--model',                              default='pointnet_model', help='model name [default: pointnet_model]')
    # parser.add_argument('--num_category',                       default=10, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--epoch',                              default=20, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate',                      default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int,                default=2048, help='Point Number')
    parser.add_argument('--optimizer', type=str,                default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str,                  default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float,             default=1e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true',   default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true',  default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample',action='store_true',default=False, help='use uniform sampiling')
    return parser.parse_args()


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def test(model, loader, num_class=2):
    classifier = model.eval()
    accuracies = []
    print("length: ", len(loader))
    
    for batch_id, (points, target) in tqdm(enumerate(loader), total=len(loader)):

        if not args.use_cpu:
            points, target = points.cuda(), target.cuda()

        points = points.transpose(2, 1)
        pred, _ = classifier(points)
        pred_choice = pred.data.max(1)[1]
        accuracies.append(accuracy_score(target.cpu(), pred_choice.cpu()))
    
    accuracy = np.mean(accuracies)
    return accuracy


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    trainDataLoader = all_train
    testDataLoader = all_test


    '''MODEL LOADING'''
    num_class = 2
    model = importlib.import_module(args.model)
    shutil.copy('./%s.py' % args.model, str(exp_dir))
    shutil.copy('./train_Pointnet.py', str(exp_dir))

    classifier = model.get_model(num_class, normal_channel=args.use_normals)
    criterion = model.get_loss()
    classifier.apply(inplace_relu)

    if not args.use_cpu:
        classifier = classifier.cuda()
        criterion = criterion.cuda()

    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    global_epoch = 0
    global_step = 0
    best_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()
        scheduler.step()

        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()
            points = points.data.numpy()
            points = torch.Tensor(points)
            points = points.transpose(2, 1)

            if not args.use_cpu:
                points, target = points.cuda(), target.cuda()

            pred, trans_feat = classifier(points)
            loss = criterion(pred, target.long(), trans_feat)
            pred_choice = pred.data.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_acc = np.mean(mean_correct)
        log_string('Train Accuracy: %f' % train_acc)

        with torch.no_grad():
            accuracy = test(classifier.eval(), testDataLoader, num_class=num_class)

            if (accuracy >= best_acc):
                best_acc = accuracy
                best_epoch = epoch + 1

            log_string(f'Test Accuracy: {accuracy}')
            log_string(f'Best Accuracy: {best_acc}')

            if (accuracy >= best_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'accuracy': accuracy,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
            global_epoch += 1

    logger.info('End of training...')


if __name__ == '__main__':
    args = parse_args()
    main(args)
