from utils.utils import to_numpy, print_results, clear
from utils.utils_metric import calc_metrics
from IPython.display import clear_output
from tqdm import tqdm
import numpy as np
import pandas as pd
import time
from torch.autograd import Variable
import torch
import os
from utils.utils_plot import vizualize
def sigmoid(x):
    return 1 / (1+np.exp(-1*x))

def validation(model, dataloaders, criterion, phase, use_gpu, device):
    model.eval()
    pred = np.array([])
    true = np.array([])
    losses = np.array([])
    with torch.no_grad():
        for data in tqdm(dataloaders[phase]):
            inputs, labels = data
            if use_gpu:
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
            else:
                inputs, labels = Variable(inputs), Variable(labels)
            outputs = model(inputs)
            loss = criterion(outputs.type(torch.cuda.FloatTensor)[:, 1],
                                 labels.type(torch.cuda.FloatTensor))
            true = np.append(true, to_numpy(labels, use_gpu))
            pred = np.append(pred, sigmoid(to_numpy(outputs[:, 1], use_gpu)))
            losses = np.append(losses, to_numpy(loss, use_gpu))
            del inputs, labels, loss
            
    model.train()
    return true, pred, np.sum(losses)


def train_model(model, criterion, optimizer, scheduler, dataloaders,
                dataset_sizes, model_path, device, fold_name,
                use_gpu=True, num_epochs=25, plot_res=True,
                predict_test=True, save_val=True):
    since = time.time()

    
    torch.save(model.state_dict(), os.path.join(model_path, '_temp'))
    best_f1 = 0.0
    best_epoch = 0
    train_metrics = []
    val_metrics = []
    
    print('=' * 10)
    print('VAL data {}'.format(fold_name))
    print('=' * 10)
    
    for epoch in range(num_epochs):
        model.train(True)

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        pred = np.array([])
        true = np.array([])

        phase = 'train'

        for data in tqdm(dataloaders[phase]):
            optimizer.zero_grad()

            inputs, labels = data
            if use_gpu:
                inputs = Variable(inputs).to(device)
                labels = Variable(labels).to(device)
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = model(inputs)

            loss = criterion(outputs.type(torch.cuda.FloatTensor)[:, 1],
                             labels.type(torch.cuda.FloatTensor))
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            true = np.append(true, to_numpy(labels, use_gpu))
            pred = np.append(pred, sigmoid(to_numpy(outputs[:, 1], use_gpu)))

        epoch_loss = running_loss / dataset_sizes[phase]
        pr_rec_auc, roc_auc, f1_05, f1_max, f1_auc, metr = calc_metrics(true, pred)
                
        val_true, val_pred, val_loss = validation(model, dataloaders,
                                                  criterion, device=device, use_gpu=use_gpu,
                                                  phase='val')

        pr_rec_auc_val, roc_auc_val, f1_05_val, f1_max_val, f1_auc_val, metr_val = calc_metrics(val_true,
                                                                  val_pred)
        if scheduler is not None:
            scheduler.step((val_loss)/dataset_sizes['val'])

        clear()
        clear_output()
        print('=' * 10)
        print('VAL data {}'.format(fold_name))
        print('=' * 10)
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        print_results(phase=phase, loss=epoch_loss,
                          roc_auc=roc_auc, pr_rec_auc=pr_rec_auc,
                          f1_max=f1_max, f1_05=f1_05, f1_auc=f1_auc)
        print('\n')
        print_results(phase='val', loss=(val_loss)/dataset_sizes['val'],
                      roc_auc=roc_auc_val, pr_rec_auc=pr_rec_auc_val,
                      f1_max=f1_max_val, f1_05=f1_05_val, f1_auc=f1_auc_val)
        train_metrics.append(metr+[epoch_loss])
        val_metrics.append(metr_val+[(val_loss)/dataset_sizes['val']])
        
        if f1_05_val > best_f1:
            best_f1 = f1_05_val
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(model_path, '_temp'))

        time_elapsed = time.time() - since
        print('Elapsed {:.0f}m {:.0f}s\n'.format(time_elapsed // 60,
              time_elapsed % 60))
        
        np.save(os.path.join(model_path, 'val_metrics_' + fold_name),
                val_metrics)
        np.save(os.path.join(model_path, 'train_metrics_' + fold_name),
                train_metrics)
        
        print('Current best: {:4f}, epoch {}'.format(best_f1, best_epoch))
        if plot_res:
            vizualize(model_path, fold_name, show=False,
                      save=True, save_format='.png')
            
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val f1_05: {:4f}'.format(best_f1))

    model.load_state_dict(torch.load(os.path.join(model_path, '_temp')))
    vizualize(model_path, fold_name, show=False, save=True, save_format='.pdf')
    
    if predict_test:
        results = pd.DataFrame()
        test_true , test_pred, _ = validation(model, dataloaders, 
                                     criterion, device=device,
                                     use_gpu=use_gpu,
                                     phase='test')
        results['pred'] = test_pred
        results['pseudo_true'] = test_true
        results['file'] = [sample[0] for sample in dataloaders['test'].dataset.samples]
        results.to_csv(os.path.join(model_path, 'test_results_' + fold_name + '.csv'),
                       index=False)
    if save_val:
        results = pd.DataFrame()
        results['true'] = val_true
        results['pred'] = val_pred
        results['file'] = [sample[0] for sample in dataloaders['val'].dataset.samples]
        results.to_csv(os.path.join(model_path, 'val_results_' + fold_name + '.csv'),
                       index=False)
    del inputs, labels, loss
    return model, best_f1
