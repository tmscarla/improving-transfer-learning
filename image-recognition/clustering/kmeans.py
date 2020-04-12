from autoencoder import *
from downloads import *
from constants import *
from utils import *
from datasets import *
import os
from clustering.cop_kmeans import cop_kmeans


def create_shifted_encoding(dataset, shift, classes_to_distort, random_perc):
    (X_train, y_train), (X_test, y_test) = load_CIFAR10()

    autoencoder = CIFAR10Autoencoder()
    path = os.path.join(ROOT_DIR, MODELS_DIR, 'autoencoders', 'CIFAR_10_autoencoder.pt')
    autoencoder.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    X_train_noisy, encoding_train = shift_AE_dataset(dataset, X_train, y_train, autoencoder=autoencoder, shift=shift,
                                                     classes_to_distort=classes_to_distort, return_encoding=True)
    X_test_noisy, encoding_test = shift_AE_dataset(dataset, X_test, y_test, autoencoder=autoencoder, shift=shift,
                                                   classes_to_distort=classes_to_distort, return_encoding=True)

    # TODO
    encoding_train, y_train = encoding_train[:5000], y_train[:5000]

    random_indices = np.random.choice(range(len(encoding_train)), int(random_perc * len(encoding_train)))

    return (encoding_train, y_train), (encoding_test, y_test), random_indices


def kmeans(dataset, shift, classes_to_distort, random_perc):
    print('Starting...')
    (encoding_train, y_train), (encoding_test, y_test),\
    random_indices = create_shifted_encoding(dataset, shift, classes_to_distort, random_perc)

    print('Encoding done!')

    encoding_known, y_known = encoding_train[random_indices], y_train[random_indices]

    must_link = []
    for i in range(len(random_indices)):
        for j in range(len(random_indices)):
            if y_train[random_indices[i]] == y_train[random_indices[j]] and i != j:
                must_link.append((random_indices[i], random_indices[j]))

    cannot_link = []
    for i in range(len(random_indices)):
        for j in range(len(random_indices)):
            if y_train[random_indices[i]] != y_train[random_indices[j]] and i != j:
                cannot_link.append((random_indices[i], random_indices[j]))

    unique_labels = len(np.unique(y_train))

    print('K-means...')

    clusters, centers = cop_kmeans(encoding_train, unique_labels, must_link, cannot_link, max_iter=100,
                                   verbose=True, initialization='random')
    mapping = {}

    for label in np.unique(y_known):
        mapping[clusters[random_indices[list(y_known).index(label)]]] = label

    y_clusters = [mapping[c].flatten() for c in clusters]
    y_clusters = [y for labels in y_clusters for y in labels]
    y_train = [y for labels in y_train for y in labels]

    correct = 0.0
    for i in range(len(y_train)):
        if y_train[i] == y_clusters[i]:
            correct += 1.0
    accuracy = correct / len(y_train)
    print(accuracy)


if __name__ == '__main__':
    kmeans(dataset=None, shift=5, classes_to_distort=None, random_perc=0.2)
