import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data as tdata
from conformal import ConformalModelLogits


def ensemble_logits(list_logits, weights):
    probs = torch.zeros_like(list_logits[0].tensors[0])  # 1000 classes in Imagenet.
    labels = list_logits[0].tensors[1]
    for i in range(len(list_logits)):
        probs += weights[i] * F.softmax(list_logits[i].tensors[0], dim=1)
    logits = torch.log(probs)

    # Construct the dataset
    dataset_logits = tdata.TensorDataset(logits, labels)
    return dataset_logits


def LACPredictionSets(logits_cal, logits_val, alpha, bsz, temp_scale=True):
    # logits_cal, logits_val= split2(logits, n_data_conf, len(logits)-n_data_conf) # A new random split for every trial
    n_data_conf = len(logits_cal)
    # Prepare the loaders
    # ground truth locations
    gt_locs_cal = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in logits_cal])
    gt_locs_val = np.array([np.where(np.argsort(x[0]).flip(dims=(0,)) == x[1])[0][0] for x in logits_val])
    # calibrate using a dummy conformal model
    loader_cal = torch.utils.data.DataLoader(logits_cal, batch_size=bsz, shuffle=False, pin_memory=True)
    if temp_scale:
        conformal_model = ConformalModelLogits(None, loader_cal, alpha=alpha, naive=True, batch_size=bsz)
        T = conformal_model.T.item()
    else:
        T = 1.0
    scores_cal = 1-np.array([np.sort(torch.softmax(logits_cal[i][0]/T, dim=0))[::-1][gt_locs_cal[i]] for i in range(len(logits_cal))])
    scores_val = 1-np.array([np.sort(torch.softmax(logits_val[i][0]/T, dim=0))[::-1][gt_locs_val[i]] for i in range(len(logits_val))])
    q = np.quantile(scores_cal, np.ceil((n_data_conf+1) * (1-alpha))/n_data_conf)
    top1_avg = (gt_locs_val == 0).mean()
    top5_avg = (gt_locs_val < 5).mean()
    cvg_avg = ( scores_val < q).mean()
    sz_avg = np.array([ ( (1-torch.softmax(logits_val[i][0]/T, dim=0)) < q).sum() for i in range(len(logits_val)) ]).mean()
    return top1_avg, top5_avg, cvg_avg, sz_avg
