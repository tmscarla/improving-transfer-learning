import torch
import numpy as np


class WeightedSampler(torch.utils.data.Sampler):

    def __init__(self, dataset, indices=None, num_samples=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # Initialize weights to 1/N
        self.weights = torch.DoubleTensor([1 / self.num_samples] * self.num_samples)

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples

    def update_weights(self, model, dataset, learning_rate):
        for idx in self.indices:
            model = model.float()

            x, y_true = dataset.__getitem__(idx)
            x = torch.from_numpy(np.array(x))

            y_hat = model(x.float())
            y_hat = torch.round(y_hat).item()

            w = self.weights[idx]
            if y_hat == y_true:
                w = w * np.exp(-learning_rate)
            else:
                w = w * np.exp(learning_rate)
            self.weights[idx] = w


