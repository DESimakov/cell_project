from utils_loader import ImageFolder
import os
from torchvision import datasets, models, transforms
import torch

data_transforms = {
    'train': transforms.Compose([
        transforms.Scale(224),
       # transforms.RandomSizedCrop(224),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Scale(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

def folds_path(train_folds, val_folds, test_data, data_dir):
    pathes = {'train': [os.path.join(data_dir, i) for i in train_folds],
     'val': [os.path.join(data_dir, i) for i in val_folds],
     'test': [os.path.join(data_dir, i) for i in test_data]}
    return pathes

def is_shuffle(stage):
    is_sh = {'train': True, 'val': False, 'test': False}    
    return is_sh

def loader(train_folds, val_folds, test_data, data_dir, bs=24):
    _pathes = folds_path(train_folds, val_folds, test_data, data_dir)
    _image_datasets = {x: ImageFolder(root=_pathes[x], transform=data_transforms[x]) for x in ['train', 'val', 'test']}
    _dataset_sizes = {x: len(_image_datasets[x]) for x in ['train', 'val', 'test']}
    _class_names = _image_datasets['train'].classes
    
    _dataloaders = {x: torch.utils.data.DataLoader(_image_datasets[x], batch_size=bs,
                                             shuffle=is_shuffle(x), num_workers=0)
              for x in ['train', 'val', 'test']}
    return _dataloaders, _dataset_sizes, _class_names    