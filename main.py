from __future__ import print_function
import argparse

from sklearn.model_selection import LeaveOneOut
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch

import pretrainedmodels
import warnings
import gc
import numpy as np
import os
warnings.filterwarnings("ignore")

from utils.utils_data_load import *
from train.train_functions import *
from utils.utils import FocalLoss
from albumentations.pytorch import ToTensor
from albumentations import (Compose, CenterCrop, VerticalFlip, RandomSizedCrop,
                            HorizontalFlip, HueSaturationValue, ShiftScaleRotate, 
                            Resize, RandomCrop, Normalize, Rotate)
'''
Dependencies:
    
albumentations==0.2.0
pretrainedmodels==0.7.4
numpy==1.16.2
pandas==0.24.1
torch==1.0.1.post2
tqdm==4.31.1
'''

def change_classes(x):
        return 1 - x
    
class BP(nn.Module):
    def __init__(self):
        super(BP, self).__init__()
        pass

    def forward(self, x):
        x = x.view(-1, 512, x.shape[2]**2)
        x = torch.bmm(x, torch.transpose(x, 1, 2))
        x = x.view(-1, 512**2)       
        return x
    
def main():

    parser = argparse.ArgumentParser(description='PyTorch Cell predict')

    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')

    parser.add_argument('--num-epochs', type=int, default=50,
                        help='number of training epochs (default: 50)')

    parser.add_argument('--model-path', default='../results/',
                        help='path to saved models (default: ../results/)')

    parser.add_argument('--data-dir', default='../data',
                        help='path to the directory with data set (default: ../data)')

    parser.add_argument('--experiment', default='exp_1',
                        help='name of the experiment (default: exp_1)')

    parser.add_argument('--description', default='',
                        help='description of the experiment (default: empty)')

    args = parser.parse_args()

    data_dir = args.data_dir
    model_path = args.model_path
    experiment = args.experiment
    description = args.description

    model_path = os.path.join(model_path, experiment)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    with open(os.path.join(model_path, 'description.txt'), 'w') as f:
        f.write("%s\n" % description)

    use_gpu = torch.cuda.is_available()

    data_transforms = {
        'train': {
                0:
        Compose([
        Rotate(15),
        CenterCrop(224, 224), 
        VerticalFlip(),
        HorizontalFlip(),
        HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=40),
        ToTensor()]),
    1:Compose([
        Rotate(15),
        CenterCrop(224, 224), 
        VerticalFlip(),
        HorizontalFlip(),
        HueSaturationValue(hue_shift_limit=50, sat_shift_limit=50, val_shift_limit=40),
        ToTensor()])},
        'val': {0: Compose([
        CenterCrop(224, 224),
        ToTensor()
    ]),
    1: Compose([
        CenterCrop(224, 224), 
        ToTensor()
    ])},
        'test': {0: Compose([
        CenterCrop(224, 224), 
        ToTensor()
    ]),
    1:Compose([
        CenterCrop(224, 224), 
        ToTensor()
    ])},
    }

    target_transform = change_classes
    loo = LeaveOneOut()

    folds = np.array(['fold_0','fold_1','fold_2'])

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for n, (tr, vl) in enumerate((list(loo.split(folds)))):
        if True:
            model_ft = pretrainedmodels.__dict__['resnet18'](num_classes=1000, pretrained='imagenet')
            model_ft.last_linear = nn.Linear(512, 2)
    
    
            if use_gpu:
                model_ft = model_ft.to(device)
            criterion = FocalLoss(gamma=0.3, alpha=None, size_average=False)
            params_to_train = model_ft.parameters()
            optimizer_ft = optim.Adam(params_to_train)
            plat_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft,
                                                                   'min', patience=3,
                                                                   factor=0.95, verbose=True)
            train_folds = folds[tr]
            val_folds = folds[vl]
            test_data = ['fold_3']
    
            dataloaders, dataset_sizes, class_names = loader(data_transforms, train_folds, val_folds,
                                                             test_data, data_dir, bs=args.batch_size,
                                                             target_transform=target_transform)
    
            model_ft, best_score = train_model(model_ft, criterion, optimizer_ft,
                                       plat_lr_scheduler,
                                       dataset_sizes=dataset_sizes,
                                       model_path=model_path,
                                       dataloaders=dataloaders,
                                       device=device, 
                                       num_epochs=args.num_epochs,
                                       fold_name=folds[vl][0], best='loss')
    
            torch.save(model_ft.state_dict(),
                       os.path.join(model_path, 'val_' + folds[vl][0] + '_f1_05_' + str(best_score).replace('.', '')))
            del criterion, optimizer_ft, plat_lr_scheduler
            torch.cuda.empty_cache()
            gc.collect()

if __name__ == '__main__':
    main()
