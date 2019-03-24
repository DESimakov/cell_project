from utils import to_numpy, print_results
from utils_metrics import calc_metrics
from tqdm import tqdm
import numpy as np
import time
from torch.autograd import Variable
import torch
import gc

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
                dataset_sizes, model_path, device, 
                use_gpu=True, num_epochs=25, verbose=1):
    since = time.time()

    
    torch.save(model.state_dict(), model_path \
               + '_temp') #model.state_dict().copy()
    best_f1_max = 0.0
    train_metrics = []
    val_metrics = []
    for epoch in range(num_epochs):
        model.train(True)

        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        running_loss = 0.0
        pred = np.array([])
        true = np.array([])

        phase = 'train'
        itr = 1

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

            if itr % verbose == 0:
                val_true, val_pred, val_loss = validation(model, dataloaders,
                                                          criterion,
                                                          device=device,
                                                          use_gpu=use_gpu,
                                                          phase='val')
                pr_rec_auc, roc_auc, f1_05, f1_max, f1_auc = \
                calc_metrics(val_true, val_pred)
                print_results('val', loss=(val_loss)/dataset_sizes['val'],
                              roc_auc=roc_auc,
                              pr_rec_auc=pr_rec_auc, f1_max=f1_max,
                              f1_05=f1_05, f1_auc=f1_auc)
                torch.save(model.state_dict(), model_path \
                           + str(epoch) + '_iter_' + str(itr) +'_f1_max_' \
                           + str(f1_max).replace('.', ''))

            itr += 1

        epoch_loss = running_loss / dataset_sizes[phase]
        pr_rec_auc, roc_auc, f1_05, f1_max, f1_auc, metr = calc_metrics(true, pred)
        print_results(phase=phase, loss=epoch_loss,
                      roc_auc=roc_auc, pr_rec_auc=pr_rec_auc,
                      f1_max=f1_max, f1_05=f1_05, f1_auc=f1_auc)
        train_metrics.append(metr+[epoch_loss])
        val_true, val_pred, val_loss = validation(model, dataloaders,
                                                  criterion, device=device, use_gpu=use_gpu,
                                                  phase='val')

        pr_rec_auc, roc_auc, f1_05, f1_max, f1_auc, metr_val = calc_metrics(val_true,
                                                                  val_pred)
        if scheduler is not None:
            scheduler.step((val_loss)/dataset_sizes['val'])
            
        print_results(phase='val', loss=(val_loss)/dataset_sizes['val'],
                      roc_auc=roc_auc, pr_rec_auc=pr_rec_auc,
                      f1_max=f1_max, f1_05=f1_05, f1_auc=f1_auc)
        val_metrics.append(metr_val+[(val_loss)/dataset_sizes['val']])
        
        if f1_05 > best_f1_max:
            best_f1_max = f1_05
            torch.save(model.state_dict(), model_path \
               +  '_temp')

        time_elapsed = time.time() - since
        print('Elapsed {:.0f}m {:.0f}s\n'.format(time_elapsed // 60,
              time_elapsed % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val f1_05: {:4f}'.format(best_f1_max))

    model.load_state_dict(torch.load(model_path \
               +  '_temp'))
    del inputs, labels, loss
    return model, best_f1_max, train_metrics, val_metrics
