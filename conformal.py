import numpy as np
import torch
import torch.nn.functional as F


def compute_critical_score(net, loader, mode="l2", alpha=0.1, device="cpu"):
    n = len(loader.dataset)
    scores = torch.zeros(n).to(device)
    count = 0
    for (i, batch) in enumerate(loader):
        inputs, targets = batch[0].to(device), batch[1].to(device)
        if mode in ["l2", "l1"]:
            scores[count:(count + inputs.shape[0])] = torch.abs(
                net(inputs).squeeze() - targets
            )
        elif mode == "quantile":
            scores[count:(count + inputs.shape[0])] = torch.max(
                targets - net(inputs)[:, 1], net(inputs)[:, 0] - targets
            )
        count += inputs.shape[0]
    critical_score = torch.sort(scores)[0][int(np.ceil((1 - alpha) * (n + 1)))]
    return critical_score


def pairwise_distances(x):
    # x should be two dimensional
    instances_norm = torch.sum(x ** 2, -1).reshape((-1, 1))
    return -2 * torch.mm(x, x.t()) + instances_norm + instances_norm.t()


def GaussianKernelMatrix(x, sigma=1):
    pairwise_distances_ = pairwise_distances(x)
    return torch.exp(-pairwise_distances_ / sigma)


def HSIC(x, y, s_x=1, s_y=1):
    m, _ = x.shape  # batch size
    K = GaussianKernelMatrix(x, s_x).float()
    L = GaussianKernelMatrix(y, s_y).float()
    H = torch.eye(m) - 1.0 / m * torch.ones((m, m))
    H = H.float().to(K.device)
    HSIC = torch.trace(torch.mm(L, torch.mm(H, torch.mm(K, H)))) / ((m - 1) ** 2)
    return HSIC

def evaluate_coverage_length(net, loader, critical_score, mode="l2", device="cpu",
                             return_corr=False):
    n = len(loader.dataset)
    coverage, length = 0., 0.
    cov_vec, len_vec = torch.zeros(len(loader.dataset)), torch.zeros(len(loader.dataset))
    count = 0
    for (i, batch) in enumerate(loader):
        inputs, targets = batch[0].to(device), batch[1].to(device)
        if mode in ["l2", "l1"]:
            coverage += torch.sum(torch.abs(targets - net(inputs).squeeze()) < critical_score)
            length += (2*critical_score) * inputs.shape[0]
        elif mode == "quantile":
            cov_vec[count:(count+inputs.shape[0])] = \
                (targets >= net(inputs)[:, 0] - critical_score) * \
                (targets <= net(inputs)[:, 1] + critical_score)
            len_vec[count:(count+inputs.shape[0])] = F.relu(
                net(inputs)[:, 1] - net(inputs)[:, 0] + 2 * critical_score
            )
            coverage += torch.sum(cov_vec[count:(count+inputs.shape[0])])
            length += torch.sum(len_vec[count:(count+inputs.shape[0])])
            count += inputs.shape[0]
    coverage /= n
    length /= n
    corr = np.corrcoef(cov_vec.detach().cpu().numpy(), len_vec.detach().cpu().numpy())[0, 1]
    hsic = HSIC(cov_vec.reshape(-1, 1), len_vec.reshape(-1, 1))
    if return_corr == False:
        return coverage, length
    return coverage, length, corr, hsic
