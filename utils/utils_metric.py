from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np


def f1_scores(y_true, y_pred, thresholds):
    eps = 1e-5
    thresholds = np.append(y_pred.min()-eps, thresholds)
    shape = thresholds.shape[0]
    f1_samples = np.zeros(shape)
    f1_05 = f1_score(y_true, (y_pred > 0.5).astype(int), average='weighted')
    # распараллелить
    for i in range(shape):
        f1_samples[i] = f1_score(y_true,
                  (y_pred > thresholds[i]).astype(int), average='weighted')

    inds = np.argsort(thresholds)
    f1_auc = auc(thresholds[inds], f1_samples[inds])
    return f1_05, f1_samples.max(), f1_auc


def calc_metrics(y_true, y_pred):
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    pr_score = precision_score(y_true, (y_pred > 0.5).astype(int),
                               average='weighted')
    rec_score = recall_score(y_true, (y_pred > 0.5).astype(int),
                               average='weighted')
    total_acc = accuracy_score(y_true, (y_pred > 0.5).astype(int))
    one_acc = accuracy_score(y_true[y_true==1],
                             (y_pred[y_true==1] > 0.5).astype(int))
    zer_acc = accuracy_score(y_true[y_true==0],
                             (y_pred[y_true==0] > 0.5).astype(int))
    pr_rec_auc = auc(recall, precision)
    roc_auc = roc_auc_score(y_true, y_pred)
    f1_05, f1_max, f1_auc = f1_scores(y_true, y_pred, thresholds)
    return (pr_rec_auc, roc_auc, f1_05, f1_max, f1_auc,
            [f1_05, pr_score, rec_score, total_acc, one_acc, zer_acc])
