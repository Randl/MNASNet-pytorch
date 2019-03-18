import torch
from torch.distributions import Beta

from utils.cross_entropy import onehot


def mixup(x, y, num_classes, gamma, smooth_eps):
    if gamma == 0 and smooth_eps == 0:
        return x, y
    m = Beta(torch.tensor([gamma]), torch.tensor([gamma]))
    lambdas = m.sample([x.size(0), 1, 1]).to(x)
    my = onehot(y, num_classes).to(x)
    true_class, false_class = 1. - smooth_eps * num_classes / (num_classes - 1), smooth_eps / (num_classes - 1)
    my = my * true_class + torch.ones_like(my) * false_class
    perm = torch.randperm(x.size(0))
    x2 = x[perm]
    y2 = my[perm]
    return x * (1 - lambdas) + x2 * lambdas, my * (1 - lambdas) + y2 * lambdas


class Mixup(torch.nn.Module):
    def __init__(self, num_classes=1000, gamma=0, smooth_eps=0):
        super(Mixup, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smooth_eps = smooth_eps

    def forward(self, input, target):
        return mixup(input, target, self.num_classes, self.gamma, self.smooth_eps)
