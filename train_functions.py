from utils import to_numpy, print_results
from utils_metrics import calc_metrics
from tqdm import tqdm
import numpy as np
import time
from torch.autograd import Variable
import torch


def validation(model, dataloaders, criterion, phase, use_gpu):
    model.eval()
    pred = np.array([])
    true = np.array([])
    losses = np.array([])
    for data in tqdm(dataloaders[phase]):
        inputs, labels = data
        if use_gpu:
            inputs = Variable(inputs.cuda())
            labels = Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)
        output = model(inputs)
        loss = criterion(output, labels)
        true = np.append(true, to_numpy(labels, use_gpu))
        pred = np.append(pred, to_numpy(output[:, 1], use_gpu))
        losses = np.append(losses, to_numpy(loss, use_gpu))
    model.train()
    return true, pred, losses


def train_model(model, criterion, optimizer, scheduler, dataloaders,
                dataset_sizes, model_path,
                use_gpu=True, num_epochs=25, verbose=1):
    since = time.time()

    best_model_wts = model.state_dict()
    best_f1_max = 0.0
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
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.data[0]

            true = np.append(true, to_numpy(labels, use_gpu))
            pred = np.append(pred, to_numpy(outputs[:, 1], use_gpu))

            if itr % verbose == 0:
                val_true, val_pred, val_loss = validation(model, dataloaders,
                                                          criterion,
                                                          use_gpu=use_gpu,
                                                          phase='val')
                pr_rec_auc, roc_auc, f1_05, f1_max, f1_auc = \
                calc_metrics(val_true, val_pred)
                print_results('val', loss=np.mean(val_loss), roc_auc=roc_auc,
                              pr_rec_auc=pr_rec_auc, f1_max=f1_max,
                              f1_05=f1_05, f1_auc=f1_auc)
                torch.save(model.state_dict(), model_path \
                           + str(epoch) + '_iter_' + str(itr) +'_f1_max_' \
                           + str(f1_max).replace('.', ''))

            itr += 1

        epoch_loss = running_loss / dataset_sizes[phase]
        pr_rec_auc, roc_auc, f1_05, f1_max, f1_auc = calc_metrics(true, pred)
        print_results(phase=phase, loss=epoch_loss,
                      roc_auc=roc_auc, pr_rec_auc=pr_rec_auc,
                      f1_max=f1_max, f1_05=f1_05, f1_auc=f1_auc)

        val_true, val_pred, val_loss = validation(model, dataloaders,
                                                  criterion, use_gpu=use_gpu,
                                                  phase='val')
        pr_rec_auc, roc_auc, f1_05, f1_max, f1_auc = calc_metrics(val_true,
                                                                  val_pred)
        print_results(phase='val', loss=np.mean(val_loss),
                      roc_auc=roc_auc, pr_rec_auc=pr_rec_auc,
                      f1_max=f1_max, f1_05=f1_05, f1_auc=f1_auc)

        if f1_max > best_f1_max:
            best_f1_max = f1_max
            best_model_wts = model.state_dict()

        time_elapsed = time.time() - since
        print('Elapsed {:.0f}m {:.0f}s\n'.format(time_elapsed // 60,
              time_elapsed % 60))

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val f1_max: {:4f}'.format(best_f1_max))

    model.load_state_dict(best_model_wts)
    return model
