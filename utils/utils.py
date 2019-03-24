from os import system, name 

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