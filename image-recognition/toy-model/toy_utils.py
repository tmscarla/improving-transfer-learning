from toy_datasets import *
import torch
import matplotlib.pyplot as plt
from toy_constants import *
import numpy as np
from toy_property import compute_samples_property
from boosting.weighted_loss import WeightedLoss
from sklearn.model_selection import StratifiedShuffleSplit


def get_data_loader(X, y, sampler=None, shuffle=True, num_workers=0):
    """
    Generate a DataLoader object given a two numpy arrays.
    :param X: numpy array of samples
    :param y: numpy array of labels
    :param shuffle: if True, shuffle samples
    :param num_workers:
    :return: a DataLoader object initialized with X and y
    """
    # y = y.flatten()  # Flatten for the DataLoader
    dataset = ToyDataset(X=X, y=y)
    return torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=BATCH_SIZE, shuffle=shuffle,
                                       num_workers=num_workers)


def abline(weight1, weight2, bias, label):
    axes = plt.gca()
    slope = -weight1 / weight2
    x_vals = np.array(axes.get_xlim())
    y_vals = -bias / weight2 + slope * x_vals
    plt.plot(x_vals, y_vals, '--', label=label)


def get_params(model):
    # Get the parameters of the model
    params = model.state_dict()
    weight1 = params['f1.weight'][0][0].numpy()
    weight2 = params['f1.weight'][0][1].numpy()
    bias = params['f1.bias'][0].numpy()
    return weight1, weight2, bias


def rotate(X, theta=45.00, ax_1=-1, ax_2=-2):
    theta = np.radians(theta)
    R = np.identity(X.shape[1])
    c, s = np.cos(theta), np.sin(theta)
    R[ax_1][ax_1], R[ax_1][ax_2], R[ax_2][ax_1], R[ax_2][ax_2] = c, s, -s, c
    X_noisy = np.array(np.matmul(X, R))
    return X_noisy


def get_samples_by_property(model, X_train_noisy, y_train, perc, most=True, prop='entropy', diversity=False,
                            min_per_class=None, rand_swap=False, rand_perc=0.2, swap_less_informative=True,
                            flatten=False):
    """
    Compute a subset of the dataset according to a property of the samples and optionally following ad-hoc heuristics.
    :param model: network that performs the classification. Samples are fed into the network to be selected according
           to their classification property
    :param X_train_noisy: numpy array of noisy samples
    :param y_train: numpy array of labels
    :param perc: relative amount of samples you want to retrieve
    :param most: select the samples with the greatest property values, otherwise the ones with the smallest
    :param prop: ['entropy', 'cross_entropy', 'probability', 'first_vs_second']
    :param diversity: apply diversity heuristic, trying to maintain a balanced number of labels in the subset
    :param min_per_class: minimum number of samples that each class must contain in the subset. If None, the treshold is
           computed ad-hoc for each class using the following heuristic: min_per_class_i = int(n_labels_i * perc)
    :type min_per_class: integer or None
    :param rand_swap: swap selected elements from the subset with random samples not selected initially
    :param rand_perc: percentage of the elements to be swept
    :param swap_less_informative: if True, swap the elements which have the least entropy (if most=True)
           or the most entropy (if most=False)
    :return: X_train_noisy_subset, y_train_noisy_subset: subset of the samples
    """
    unique_labels = len(np.unique(y_train))
    indices = compute_samples_property(model, X_train_noisy, y_train, unique_labels, prop, indices=True,
                                       flatten=flatten)
    n_samples = int(len(indices) * perc)

    if most:
        indices = indices[::-1]

    indices_subset = indices[:n_samples]

    if rand_swap:
        assert 0 <= rand_perc <= 1
        n_swap_elements = int(len(indices_subset) * rand_perc)

        if swap_less_informative:
            indices_removed = indices_subset[-n_swap_elements:]
        else:
            indices_removed = np.random.choice(indices_subset, size=n_swap_elements, replace=False)
        not_selected = np.setdiff1d(np.arange(len(indices)), indices_subset)
        indices_added = np.random.choice(not_selected, size=n_swap_elements, replace=False)

        # Remove old indices and concatenate new indices
        indices_subset = np.concatenate((np.setdiff1d(indices_subset, indices_removed), indices_added))

    X_train_noisy_subset = X_train_noisy[indices_subset]
    y_train_subset = y_train[indices_subset]
    return X_train_noisy_subset, y_train_subset


def get_random_subset(x_train_noisy, y_train, entropy_percentage, return_indices=False):
    total_len = x_train_noisy.shape[0]
    subset_len = int(total_len * entropy_percentage)
    indices = np.random.choice(total_len, subset_len, replace=False)
    x_train_noisy_subset = x_train_noisy[indices]
    y_train_subset = y_train[indices]

    if return_indices:
        return indices
    else:
        return x_train_noisy_subset, y_train_subset


def init_exponential_loss(X):
    weights_boosting = dict()
    for sample in X:
        weights_boosting[tuple(sample)] = 1 / len(X)
    criterion = WeightedLoss(X=X, weights_boosting=weights_boosting,
                                        indices=list(range(len(X))))
    return criterion, weights_boosting


def dataset_split(X, y, perc=VALIDATION_PERCENTAGE, random_state=RANDOM_SEED, return_data='samples'):
    """
    Given two arrays of samples and label X and y, perform a random splitting in train and validation sets.
    :param X: numpy array of samples
    :param y: numpy array of labels
    :param val_perc: percentage of validation set
    :param random_state: random state of the splitter
    :param return_data: if True, return DataLoader objects instead of numpy arrays
    :return: (train_loader, val_loader) or (X_train, y_train), (X_val, y_val) or train_idx, val_idx
    """
    assert 0 <= perc <= 1

    sss = StratifiedShuffleSplit(n_splits=1, test_size=perc, random_state=random_state)
    train_idxs, valid_idxs = next(sss.split(X, y))

    X_train, X_valid = X[train_idxs], X[valid_idxs]
    y_train, y_valid = y[train_idxs], y[valid_idxs]

    if return_data == 'data_loader':
        return get_data_loader(X_train, y_train), get_data_loader(X_valid, y_valid)
    elif return_data == 'samples':
        return (X_train, y_train), (X_valid, y_valid)
    elif return_data == 'indices':
        return train_idxs, valid_idxs


def convert_labels(y, old_labels, new_labels):
    """
    Convert labels of a numpy array using an implicit mapping from old_labels to new_labels.
    :param y: numpy array of labels
    :param old_labels: e.g. [0, 1]
    :param new_labels: e.g. [-1, 1]
    :return: y_converted
    """
    assert len(old_labels) == len(new_labels)
    if set(y) == set(new_labels):
        return y
    mapping = dict(zip(old_labels, new_labels))
    y_converted = [mapping[y_i] for y_i in y]
    return np.array(y_converted)


def evaluate_ensemble(ensemble, learning_rates, dataloader, device):
    """
    Evaluate ensamble of models.
    :param ensemble:
    :param learning_rates:
    :param dataloader:
    :return:
    """
    total = 0.0
    correct = 0.0
    with torch.no_grad():
        for data in dataloader:
            # Unwrap tuple
            input, labels = data
            # Move to GPU
            input, labels = input.to(device), labels.to(device)
            predicted = [0.0] * input.shape[0]
            for i, model in enumerate(ensemble):
                # Forward pass
                outputs = model(input)
                predicted = predicted + np.array(learning_rates[i]) * outputs.view(-1).tolist()
            predicted = np.where(predicted > 0, 1, -1)
            predicted = torch.FloatTensor(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # Return accuracy
    return correct / total
