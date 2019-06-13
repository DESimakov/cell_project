from os import system, name 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def to_numpy(grad_tensor, use_gpu):
    if use_gpu:
        return grad_tensor.data.cpu().numpy().flatten()
    else:
        return grad_tensor.data.numpy().flatten()

def print_results(phase, loss, pr_rec_auc, roc_auc, f1_05, f1_max, f1_auc):
    print('{} Loss: {:.4f}, roc_auc: {:.4f}, pr_rec_auc: {:.4f}, f1_max: {:.4f}, f1-05: {:.4f}, f1_auc: {:.4f}'.format(
                phase, loss, roc_auc, pr_rec_auc, f1_max, f1_05, f1_auc))

def clear(): 
    if name == 'nt': 
        _ = system('cls') 
    else: 
        _ = system('clear')
        

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        #target = target.view(-1,1)

        logpt = torch.log(input.sigmoid())
        #logpt = logpt.gather(1,target)
        #logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())
        #print(pt)
        C = (1-pt)**self.gamma
        cond = (target.data == 1).type(torch.cuda.FloatTensor)
        #C = cond * C + (1 - cond) * torch.ones(input.shape[0]).cuda()
        loss =  C*F.binary_cross_entropy_with_logits(input, target,
                                                      reduce=False,
               #pos_weight=torch.Tensor([0.2]).cuda() 
               )
        if self.size_average: return loss.mean()
        else: return loss.sum()