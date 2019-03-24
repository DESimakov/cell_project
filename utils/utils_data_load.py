from utils.utils_loader import ImageFolder
import os
import torch

def folds_path(train_folds, val_folds, test_data, data_dir):
    pathes = {'train': [os.path.join(data_dir, i) for i in train_folds],
     'val': [os.path.join(data_dir, i) for i in val_folds],
     'test': [os.path.join(data_dir, i) for i in test_data]}
    return pathes

def is_shuffle(stage):
    is_sh = {'train': True, 'val': False, 'test': False}    
    return is_sh

def loader(data_transforms, train_folds, val_folds, test_data, data_dir, bs=24,
           target_transform=None):
    _pathes = folds_path(train_folds, val_folds, test_data, data_dir)
    _image_datasets = {x: ImageFolder(root=_pathes[x], transform=data_transforms[x],
                                      target_transform=target_transform) for x in ['train', 'val', 'test']}
    _dataset_sizes = {x: len(_image_datasets[x]) for x in ['train', 'val', 'test']}
    _class_names = _image_datasets['train'].classes
    _dataloaders = {x: torch.utils.data.DataLoader(_image_datasets[x], batch_size=bs,
                                             shuffle=is_shuffle(x), num_workers=0)
        for x in ['train', 'val', 'test']}
    return _dataloaders, _dataset_sizes, _class_names    