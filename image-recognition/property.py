from scipy.stats import entropy
from datasets import *
from models import *
from constants import *
from collections import Counter
from sklearn.metrics import log_loss
import warnings


def get_samples_by_property(model, X_train_noisy, y_train, perc, most=True, prop='entropy', diversity=False,
                            min_per_class=None, rand_swap=False, rand_perc=0.2, swap_less_informative=True, flatten=False):
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
    indices = compute_samples_property(model, X_train_noisy, y_train, unique_labels, prop, indices=True, flatten=flatten)
    n_samples = int(len(indices) * perc)

    if most:
        indices = indices[::-1]

    if diversity:
        return diversity_rearrangement(X_train_noisy[indices], y_train[indices],
                                       perc, min_per_class=min_per_class)

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


def diversity_rearrangement(X_train_noisy_ordered, y_train_ordered, perc, min_per_class):
    """
    Rearrange X_train and y_train keeping diversity over classes.
    :param X_train_noisy_ordered: samples ordered by some property
    :param y_train_ordered: labels ordered by some property
    :param perc: perc of samples to be kept inside
    :param min_per_class: minimum number of samples for each class to be kept
    :return: X_train_sub, y_train_sub: subset of the training set to be kept
    """
    assert 0 <= perc <= 1
    assert len(X_train_noisy_ordered) == len(y_train_ordered)

    # Samples to keep = len * perc
    keep_n_samples = int(len(X_train_noisy_ordered) * perc)
    X_new = np.copy(X_train_noisy_ordered)
    y_new = np.copy(y_train_ordered)

    # If None, each class must contain a number of values equals to the number of elements in the class * perc
    if min_per_class is None:
        min_per_class = {k: int(v*perc) for k, v in Counter(y_new).items()}
    # Adjust min_per_class if it is greater than the number of elements of the least populated class
    elif (min(Counter(y_new).values())) < min_per_class:
        min_per_class = {k: min(Counter(y_new).values()) for k, v in Counter(y_new).items()}

    # Initialization
    labels_above = []
    labels_under = []
    labels_dict = dict(Counter(y_train_ordered[:keep_n_samples]))

    for label, n_samples in labels_dict.items():
        if n_samples < min_per_class[label]:
            labels_under.append(label)
        else:
            labels_above.append(label)
    swap_i = keep_n_samples

    # Loop until diversity is reached
    for i in range(keep_n_samples - 1, -1, -1):
        if y_new[i] in labels_above:
            while y_new[swap_i] not in labels_under:
                swap_i += 1
                # Return if no further modifications can be made
                if swap_i >= len(X_new):
                    return X_new[:keep_n_samples], y_new[:keep_n_samples]

            # Adjust labels count and support lists
            labels_dict[y_new[i]] -= 1
            labels_dict[y_new[swap_i]] += 1
            if labels_dict[y_new[i]] <= min_per_class[y_new[i]]:
                labels_above.remove(y_new[i])
            if labels_dict[y_new[swap_i]] >= min_per_class[y_new[swap_i]]:
                labels_under.remove(y_new[swap_i])

            # Finally swap values
            X_new[i], X_new[swap_i] = X_new[swap_i], X_new[i]
            y_new[i], y_new[swap_i] = y_new[swap_i], y_new[i]

        if not labels_under:
            break

    return X_new[:keep_n_samples], y_new[:keep_n_samples]


def compute_samples_property(model, X_train_noisy, y_train, unique_labels, prop, indices=True,
                             flatten=False, verbose=False):
    """
    Compute a selected property for each sample of X_train_noisy with respect to model.
    :param model: pre-trained neural network
    :param X_train_noisy: numpy array of noisy samples
    :param y_train: numpy array of labels
    :param unique_labels: complete list or numpy array of unique labels used by the model. If it is an int,
           labels are assumed to be in range(0, unique_labels)
    :param prop: ['entropy', 'cross_entropy', 'probability', 'first_vs_second']
    :param indices: return indices instead of values
    :return: values or indices in ascending order
    """
    # Check unique_labels
    if type(unique_labels) is int:
        unique_labels = list(range(0, unique_labels))

    # Disable dropout
    model.eval()
    model.to(device)
    values = []

    X_train_noisy_scaled, _, _ = image_preprocessing(X_train_noisy, scale_only=False)
    dataset_noisy = WrapperDataset(X=X_train_noisy_scaled, y=y_train)
    dataloader_noisy = torch.utils.data.DataLoader(dataset_noisy, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for data in tqdm(dataloader_noisy, desc='Computing {}'.format(prop), disable=not verbose):

            # Unwrap tuple
            images, labels = data

            if flatten:
                images = images.view(images.shape[0], -1)

            # Move to GPU
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)

            # Normalization of output
            for i in range(len(outputs)):
                if prop == 'entropy':
                    v = entropy(outputs[i].cpu().numpy())
                elif prop == 'cross_entropy':
                    pred = outputs[i].cpu().numpy()
                    if len(pred) == 1:
                        pred = np.append(pred, 1-pred[0])
                    pred = pred.reshape(1, len(unique_labels))
                    label = labels[i].cpu().numpy().reshape(1, 1)
                    v = log_loss(label, pred, labels=unique_labels)
                elif prop == 'probability':
                    v = max(outputs[i].cpu().numpy().reshape(-1, 1).squeeze())
                elif prop == 'first_vs_second':
                    ordered_prob = np.sort(outputs[i].cpu().numpy().reshape(-1, 1).squeeze())[::-1]
                    v = ordered_prob[0] - ordered_prob[1]
                values.append(v)
    if indices:
        return np.argsort(values)
    else:
        return values


def cross_entropy(targets, predictions, epsilon=1e-12):
    """
    Computes cross entropy between targets (encoded as one-hot vectors)
    and predictions.
    Input: predictions (N, k) ndarray
           targets (N, k) ndarray
    Returns: scalar
    """
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets * np.log(predictions + 1e-9)) / N
    return ce
