from torch.nn.modules.loss import BCELoss
import torch
from torch.nn import functional as F
from constants import BATCH_SIZE


class WeightedLoss(BCELoss):
    def __init__(self, X, weights_boosting, indices=None, weight=None, size_average=None, reduce=None,
                 reduction='mean', batch_size=BATCH_SIZE, loss='cross'):
        super(BCELoss, self).__init__(weight, size_average, reduce, reduction)
        if loss not in ['exp', 'cross']:
            raise ValueError('Invalid choice of loss')
        self.batch_size = batch_size
        self.current_batch = 0
        self.indices = indices
        self.X = X
        self.weights_boosting = weights_boosting
        self.loss = loss

    def forward(self, prediction, target):
        total_errors = torch.zeros(1)

        for i, sample in enumerate(prediction):
            y_hat = prediction[i]
            y = target[i]
            if self.loss == 'exp':
                error = torch.exp(-(y_hat * y))
            elif self.loss == 'cross':
                y_hat, y = y_hat.unsqueeze(dim=0), y.unsqueeze(dim=0)
                if y_hat.shape[1] <= 1:
                    error = torch.nn.functional.binary_cross_entropy(y_hat, y)
                else:
                    error = torch.nn.functional.cross_entropy(y_hat, y)
            error = error * self.weights_boosting[tuple(self.X[i].flatten())]
            total_errors = total_errors.cpu() + error.cpu()

        return total_errors


def init_loss(X, loss='exp'):
    weights_boosting = dict()
    for sample in X:
        weights_boosting[tuple(sample.flatten())] = 1 / len(X)
    criterion = WeightedLoss(X=X, weights_boosting=weights_boosting,
                                        indices=list(range(len(X))), loss=loss)
    return criterion, weights_boosting
