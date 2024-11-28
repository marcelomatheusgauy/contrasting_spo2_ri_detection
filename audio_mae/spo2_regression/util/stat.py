import numpy as np
from scipy import stats
from sklearn import metrics
from scipy.stats import pearsonr
import torch

def d_prime(auc):
    standard_normal = stats.norm()
    d_prime = standard_normal.ppf(auc) * np.sqrt(2.0)
    return d_prime

@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output

def calculate_stats(output, target):
    """Calculate statistics including mAP, AUC, etc.

    Args:
      output: 2d array, (samples_num, classes_num)
      target: 2d array, (samples_num, classes_num)

    Returns:
      stats: list of statistic of each class.
    """

    #classes_num = target.shape[-1]
    stats = {}

    # Accuracy, only used for single-label classification such as esc-50, not for multiple label one such as AudioSet
    #acc = metrics.accuracy_score(np.argmax(target, 1), np.argmax(output, 1))
    mse_error = metrics.mean_squared_error(target[:,0], output[:,0])
    mae_error = metrics.mean_absolute_error(target[:,0], output[:,0])
    std_error = np.std(np.abs(target[:,0]-output[:,0]))
    r2 = metrics.r2_score(target[:,0], output[:,0])
    pearson, _ = pearsonr(target[:,0],output[:,0])
    for res in ["Saida Fold: {}  Alvo Fold: {}".format(x,y) for x,y in zip(output,target)] :
         print(res)
    print("MSE error", mse_error)
    print("Error Avg:", mae_error)
    print("Std error:",  std_error)
    print("R2:", r2)
    print("Pearson:", pearson)
    
    stats["mse_error"] = mse_error
    stats["mae_error"] = mae_error
    stats["std_error"] = std_error
    stats["r2"] = r2
    stats["pearson"] = pearson

    return stats
