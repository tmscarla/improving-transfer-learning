from sklearn.preprocessing import StandardScaler
from toy_datasets import *
import torch
from scipy.stats import entropy
from toy_constants import *


def compute_samples_property(model, X_train_noisy, y_train, unique_labels, prop, indices=True, flatten=False):
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
        unique_labels = range(0, unique_labels)

    # Disable dropout
    model.eval()
    model.to(device)
    values = []

    X_train_noisy_scaled = StandardScaler().fit_transform(X_train_noisy)
    dataset_noisy = ToyDataset(X=X_train_noisy_scaled, y=y_train)
    dataloader_noisy = torch.utils.data.DataLoader(dataset_noisy, batch_size=BATCH_SIZE, shuffle=False)

    with torch.no_grad():
        for data in dataloader_noisy:

            # Unwrap tuple
            inputs, labels = data

            if flatten:
                inputs = inputs.view(inputs.shape[0], -1)

            # Move to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)

            # Normalization of output
            for i in range(len(outputs)):
                if prop == 'entropy':
                    pred = outputs[i].cpu().numpy()[0]
                    # Handle [-1, 1] labels
                    if -1 in unique_labels:
                        pred = (pred + 1) / 2
                    v = entropy(np.array([pred, 1 - pred]))
                elif prop == 'cross_entropy':
                    label = labels[i].cpu().numpy().reshape(1, 1)[0]
                    prob = outputs[i].cpu().numpy().reshape(1, 1)[0]
                    v = cross_entropy_(prob, label)
                elif prop == 'probability':
                    v = outputs[i].cpu().numpy().reshape(1, 1)[0][0]
                elif prop == 'first_vs_second':
                    ordered_prob = np.sort(outputs[i].cpu().numpy().reshape(-1, 1).squeeze())[::-1]
                    v = ordered_prob[1] - ordered_prob[0]
                values.append(v)
    if indices:
        return np.argsort(values)
    else:
        return values


def cross_entropy_(y_hat, y):
    if y == 1:
        return -np.log(y_hat + 0.0000000001)[0]
    else:
        return -np.log(1 - y_hat + 0.0000000001)[0]
