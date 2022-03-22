import torch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def FreezeReps(net):
    """Freezes representation layers of a MLP."""
    for layer in net.layers:
        layer.weight.requires_grad = False
        try:
            layer.bias.requires_grad = False
        except:
            continue

