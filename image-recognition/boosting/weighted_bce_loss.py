from torch.nn.modules.loss import BCELoss
import torch
from torch.nn import functional as F


class WeightedBCELoss(BCELoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction='mean', weights_boosting=None):
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)
        self.weights_boosting = torch.Tensor(weights_boosting)

    def forward(self, input, target):

        total_cross_entropies = torch.zeros(1)

        for i, sample in enumerate(input):
            y_hat = input[i]
            y = target[i]
            c_e = F.binary_cross_entropy(y_hat, y, weight=self.weight,
                                         reduction=self.reduction)
            c_e = c_e * self.weights_boosting[i]
            total_cross_entropies = total_cross_entropies + c_e

        return total_cross_entropies

