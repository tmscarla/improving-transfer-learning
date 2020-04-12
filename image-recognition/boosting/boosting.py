import torch
import numpy as np
import copy
from matplotlib import pyplot as plt
from scipy.stats import entropy



# def update_weights_boosting(model, weights_boosting, dataloader, learning_rate_collection,
#                             epsilon_collection, device, lr=None):
#     """
#     :param model: neural network model
#     :param weights_boosting: dic
#     :param X: numpy array of samples
#     :param y: numpy array of labels
#     :param learning_rate_collection:
#     :param epsilon_collection:
#     :param device:
#     :param lr:
#     :return:
#     """
#
#     model.eval()
#     model = model.to(device)
#
#     # Compute epsilon for weigths update
#     epsilon = 0.0
#     wrong = 0
#
#     with torch.no_grad():
#         for data in dataloader:
#             image, label = data
#
#             # Move to GPU
#             image, label = image.to(device), label.to(device)
#
#             # Forward pass
#             output = model(image)
#             output = output.cpu()
#
#             predicted = torch.where(output.data > 0, torch.ones(output.data.shape),
#                                     torch.ones(output.data.shape) * (-1))
#
#             label = label.float()
#             output.view(-1)
#
#             predicted = predicted.view(-1)
#             predicted = predicted.to(device)
#
#             w = weights_boosting[tuple(image.cpu().detach().numpy().flatten())]
#
#             if label != predicted:
#                 wrong += 1
#                 epsilon = epsilon + w
#
#     print(wrong)
#     epsilon = epsilon / sum(weights_boosting.values())
#     #print(epsilon)
#
#     model.train()
#
#     return weights_boosting, epsilon, learning_rate_collection, epsilon_collection
from tqdm import tqdm


def update_weights_boosting(model, weights_boosting, X, y, device, learning_rate_collection=[],
                            epsilon_collection=[], lr=None, loss='cross', flatten=False, mode='normal'):
    """
    :param model: neural network model
    :param weights_boosting: dic
    :param X: numpy array of samples
    :param y: numpy array of labels
    :param learning_rate_collection:
    :param epsilon_collection:
    :param device:
    :param lr:
    :return:
    """

    model.eval()
    model = model.to(device)

    # Compute epsilon for weigths update
    epsilon = 0.0
    wrong = 0

    for idx in range(len(X)):
        x, y_true = X[idx], y[idx]
        x = copy.deepcopy(x)
        if flatten:
            x = x.flatten()
        x = np.moveaxis(x, source=-1, destination=0).astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x)
        x = x.to(device)

        y_hat = model(x.float()).cpu()

        if loss == 'exp':
            y_hat = torch.where(y_hat > 0, torch.ones(1), torch.ones(1) * (-1))
        else:
            _, y_hat = torch.max(y_hat.data, 1)

        w = weights_boosting[tuple(X[idx].flatten())]

        if y_hat != y_true:
            wrong += 1
            epsilon = epsilon + w

    epsilon = epsilon / sum(weights_boosting.values())

    if lr is not None:
        learning_rate = lr
    else:
        learning_rate = 0.5 * np.log((1 - epsilon) / epsilon)

    if learning_rate_collection is not None:
        learning_rate_collection.append(learning_rate)
    if epsilon_collection is not None:
        epsilon_collection.append(epsilon)

    # Update weights for boosting
    for idx in tqdm(range(len(X)), desc='Updating weights'):
        x, y_true = X[idx], y[idx]
        x = copy.deepcopy(x)
        if flatten:
            x = x.flatten()
        x = np.moveaxis(x, source=-1, destination=0).astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x)
        x = x.to(device)

        y_hat = model(x.float()).cpu()

        if mode == 'normal':
            if loss == 'exp':
                y_hat = torch.where(y_hat > 0, torch.ones(1), torch.ones(1) * (-1))
            else:
                _, y_hat = torch.max(y_hat.data, 1)

        w = weights_boosting[tuple(X[idx].flatten())]

        if mode == 'normal':
            if y_hat == y_true:
                w = w * np.exp(-learning_rate)
            else:
                w = w * np.exp(learning_rate)

        elif mode == 'entropy':
            y_hat = y_hat.detach().numpy()
            if y_hat.shape[1] == 1:
                y_hat = np.append(y_hat, 1 - y_hat[0])
            w = entropy(y_hat[0])

        weights_boosting[tuple(X[idx].flatten())] = w

    # Normalization of weights
    sum_weights = sum(weights_boosting.values())
    for k, weight in weights_boosting.items():
        weights_boosting[k] = weight/sum_weights

    model.train()

    return weights_boosting
