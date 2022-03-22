import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim=1, bias=True):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=bias)

    def forward(self, x):
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden_dim=64, depth=3, bias=True,
                 freeze_reps=False, dropout=0.0):
        super(MLP, self).__init__()
        d_in, d_out = in_dim, hidden_dim
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for i in range(depth - 1):
            layer = nn.Linear(d_in, d_out, bias=bias)
            if freeze_reps:
                layer.weight.requires_grad = False
                if bias:
                    layer.bias.requires_grad = False
            self.layers.append(layer)
            self.dropouts.append(nn.Dropout(dropout))
            d_in = d_out
            d_out = hidden_dim

        self.linear = nn.Linear(d_in, out_dim, bias=bias)
        self.dropout = dropout

    def forward(self, x):
        out = x
        for (layer, dropout) in zip(self.layers, self.dropouts):
            out = layer(out)
            out = F.relu(out)
            if self.dropout > 0:
                out = dropout(out)
        return self.linear(out)


class PinballLoss():
    """Pinball loss for quantile regression"""

    def __init__(self, quantile=0.10, reduction='none'):
        self.quantile = quantile
        assert 0 < self.quantile < 1
        self.reduction = reduction

    def __call__(self, output, target):
        assert output.shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)
        error = target - output
        smaller_index = error < 0
        bigger_index = 0 < error
        loss[smaller_index] = (1 - self.quantile) * (abs(error)[smaller_index])
        loss[bigger_index] = self.quantile * (abs(error)[bigger_index])

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


class TwoSidedPinballLoss():
    """Two-sided Pinball loss for quantile regression"""

    def __init__(self, quantile_lo=0.05, quantile_hi=0.95, reduction='none'):
        self.quantile_lo, self.quantile_hi = quantile_lo, quantile_hi
        assert 0 < self.quantile_lo < self.quantile_hi < 1
        self.reduction = reduction

    def __call__(self, output, target):
        assert output.shape[-1] == 2 and output[:, 0].shape == target.shape
        loss = torch.zeros_like(target, dtype=torch.float)

        error_lo = target - output[:, 0]
        smaller_index = error_lo < 0
        bigger_index = 0 < error_lo
        loss[smaller_index] += (1 - self.quantile_lo) * (abs(error_lo)[smaller_index])
        loss[bigger_index] += self.quantile_lo * (abs(error_lo)[bigger_index])

        error_hi = target - output[:, 1]
        smaller_index = error_hi < 0
        bigger_index = 0 < error_hi
        loss[smaller_index] += (1 - self.quantile_hi) * (abs(error_hi)[smaller_index])
        loss[bigger_index] += self.quantile_hi * (abs(error_hi)[bigger_index])

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss