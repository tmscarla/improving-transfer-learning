import torchvision.datasets as dsets
import torchvision.transforms as transforms
import os
import sys
from six.moves import cPickle
import numpy as np
from constants import *
import os
import time
import torch
import requests
import pandas as pd
import cv2


def load_dataset(dataset):
    """
    Load a dataset given its name.
    :param dataset: dataset name
    :return: (X_train, y_train), (X_test, y_test): numpy arrays of samples and labels
    """

    if dataset not in DSETS:
        raise RuntimeError("Dataset not in list")
    elif dataset == 'CIFAR_10':
        (X_train, y_train), (X_test, y_test) = load_CIFAR10()
    elif dataset == 'CIFAR_100':
        (X_train, y_train), (X_test, y_test) = load_CIFAR100()
    elif dataset == 'MNIST':
        (X_train, y_train), (X_test, y_test) = load_MNIST()
    elif dataset == 'EMNIST':
        (X_train, y_train), (X_test, y_test) = load_EMNIST()
    elif dataset == 'USPS':
        (X_train, y_train), (X_test, y_test) = load_USPS()
    else:
        raise Exception()

    return (X_train, y_train), (X_test, y_test)


def download_MNIST():
    """
    Download MNIST dataset and save it into data folder.
    """
    data_folder = os.path.join(ROOT_DIR, DATA_DIR)

    dsets.MNIST(root=data_folder, train=True, transform=transforms.ToTensor(), download=True)
    dsets.MNIST(root=data_folder, train=False, transform=transforms.ToTensor())


def download_EMNIST(split='letters'):
    """
    Download EMNIST dataset and save it into data folder.
    :param split: ['balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist']
    """
    data_folder = os.path.join(ROOT_DIR, DATA_DIR)

    dsets.EMNIST(root=data_folder, train=True, transform=transforms.ToTensor(), download=True, split=split)
    dsets.EMNIST(root=data_folder, train=False, transform=transforms.ToTensor(), download=True, split=split)


def download_CIFAR(dataset='CIFAR_10', train=True):
    """
    Download CIFAR dataset and save it into data folder.
    :param dataset: type of the CIFAR dataset to be downloaded
    :type dataset: ['CIFAR10', 'CIFAR100']
    :param train: if True, train is downloaded. If False, test is downloaded
    """
    data_folder = os.path.join(ROOT_DIR, DATA_DIR)

    if dataset == 'CIFAR_10':
        if train:
            return dsets.CIFAR10(root=data_folder,
                                 train=True,
                                 transform=transforms.ToTensor(),
                                 download=True)
        else:
            return dsets.CIFAR10(root=data_folder,
                                 train=False,
                                 transform=transforms.ToTensor(),
                                 download=True)

    elif dataset == 'CIFAR_100':
        if train:
            return dsets.CIFAR100(root=data_folder,
                                  train=True,
                                  transform=transforms.ToTensor(),
                                  download=True)
        else:
            return dsets.CIFAR10(root=data_folder,
                                 train=False,
                                 transform=transforms.ToTensor(),
                                 download=True)

    # Removing tar.gz archives
    if os.path.exists(os.path.join(data_folder, 'cifar-10-python.tar.gz')):
        time.sleep(2)
        os.remove(os.path.join(data_folder, 'cifar-10-python.tar.gz'))
    if os.path.exists(os.path.join(data_folder, 'cifar-100-python.tar.gz')):
        time.sleep(2)
        os.remove(os.path.join(data_folder, 'cifar-100-python.tar.gz'))

    else:
        raise ValueError("dataset must be 'CIFAR_10' or 'CIFAR_100'")


def load_batch(file_path, label_key='labels'):
    """
    Internal utility for parsing CIFAR data.
    :param file_path: path the file to parse.
    :param label_key: key for label data in the retrieve dictionary
    :return (data, labels)
    """
    with open(file_path, 'rb') as f:
        if sys.version_info < (3,):
            d = cPickle.load(f)
        else:
            d = cPickle.load(f, encoding='bytes')
            # Decode UTF-8
            d_decoded = {}
            for k, v in d.items():
                d_decoded[k.decode('utf8')] = v
            d = d_decoded
    data = d['data']
    labels = d[label_key]

    data = data.reshape(data.shape[0], 3, 32, 32)
    return data, labels


def load_CIFAR10(channel_last=True):
    """
    Loads CIFAR10 dataset.
    :param channel_last: number of channels as last element of the tensor
    :return (x_train, y_train), (x_test, y_test)
    """
    dirname = 'cifar-10-batches-py'
    path = os.path.join(ROOT_DIR, 'data', dirname)

    # Check if dataset was previously downloaded
    if not os.path.exists(os.path.join(ROOT_DIR, DATA_DIR, dirname)):
        print('Downloading CIFAR_10 dataset...', sep='')
        download_CIFAR(dataset='CIFAR_10', train=True)
        download_CIFAR(dataset='CIFAR_10', train=False)
        print('done!')
        time.sleep(3)

    num_train_samples = 50000

    x_train = np.empty((num_train_samples, 3, 32, 32), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if channel_last:
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def load_CIFAR100(channel_last=True, label_mode='fine'):
    """
    Loads CIFAR100 dataset.
    :param channel_last: number of channels as last element of the tensor
    :param label_mode: if coarse, superclasses as used as labels, if fine, subclasses
    :type: ['fine', 'coarse']
    :return (x_train, y_train), (x_test, y_test)
    """
    assert label_mode in ['fine', 'coarse']

    dirname = 'cifar-100-python'
    origin = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
    file_name = 'cifar-100-python.tar.gz'

    # Check if dataset was previously downloaded
    if not os.path.exists(os.path.join(ROOT_DIR, DATA_DIR, dirname)):
        print('Downloading CIFAR_100 dataset...', sep='')
        download_CIFAR(dataset='CIFAR_100', train=True)
        download_CIFAR(dataset='CIFAR_100', train=False)
        time.sleep(3)

    # Load train and test and generate numpy arrays
    fpath = os.path.join(os.path.join(ROOT_DIR, DATA_DIR, dirname), 'train')
    x_train, y_train = load_batch(fpath, label_key=label_mode + '_labels')

    fpath = os.path.join(os.path.join(ROOT_DIR, DATA_DIR, dirname), 'test')
    x_test, y_test = load_batch(fpath, label_key=label_mode + '_labels')

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if channel_last:
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)


def load_MNIST():
    """
    Load MNIST dataset
    :return: (x_train, y_train), (x_test, y_test)
    """
    dirname = 'processed'
    dataset = 'MNIST'

    if not os.path.exists(os.path.join(ROOT_DIR, DATA_DIR, dataset, dirname)):
        print('Downloading MNIST dataset...', sep='')
        download_MNIST()
        time.sleep(3)

    X_train, y_train = torch.load(os.path.join(ROOT_DIR, DATA_DIR, dataset, dirname, 'training.pt'))
    X_test, y_test = torch.load(os.path.join(ROOT_DIR, DATA_DIR, dataset, dirname, 'test.pt'))

    X_train = X_train.numpy()
    y_train = y_train.numpy()
    X_test = X_test.numpy()
    y_test = y_test.numpy()

    return (X_train, y_train), (X_test, y_test)


def load_EMNIST(split='letters'):
    """
    Load EMNIST dataset
    :param split: ['balanced', 'byclass', 'bymerge', 'digits', 'letters', 'mnist']
    :return: (x_train, y_train), (x_test, y_test)
    """
    dirname = 'processed'
    dataset = 'EMNIST'

    if not os.path.exists(os.path.join(ROOT_DIR, DATA_DIR, dataset, dirname)):
        print('Downloading EMNIST dataset...', end='')
        download_EMNIST(split=split)
        print('done!')
        time.sleep(3)

    print('Loading EMNIST-{}...'.format(split), end='')
    X_train, y_train = torch.load(os.path.join(ROOT_DIR, DATA_DIR, dataset, dirname, 'training_{}.pt'.format(split)))
    X_test, y_test = torch.load(os.path.join(ROOT_DIR, DATA_DIR, dataset, dirname, 'test_{}.pt'.format(split)))

    X_train = X_train.numpy()
    y_train = y_train.numpy()
    X_test = X_test.numpy()
    y_test = y_test.numpy()
    print('done!')

    return (X_train, y_train), (X_test, y_test)


def load_USPS(resize=(28, 28)):
    """
    Download and load USPS dataset.
    :param resize: (w, h) resize each image of the dataset
    :type resize: tuple or None
    :return: (X_train, y_train), (X_test, y_test)
    """
    url_train = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.train.gz"
    url_test = "https://web.stanford.edu/~hastie/ElemStatLearn/datasets/zip.test.gz"

    # Download
    if not os.path.exists(ROOT_DIR + '/' + DATA_DIR + '/USPS/train.gz'):
        print('Downloading USPS dataset...', end='')
        os.mkdir(os.path.join(ROOT_DIR, DATA_DIR, 'USPS'))

        # Train
        filename = os.path.join(ROOT_DIR, DATA_DIR, 'USPS', 'train.gz')
        with open(filename, "wb") as f:
            r = requests.get(url_train)
            f.write(r.content)

        # Test
        filename = os.path.join(ROOT_DIR, DATA_DIR, 'USPS', 'test.gz')
        with open(filename, "wb") as f:
            r = requests.get(url_test)
            f.write(r.content)

        time.sleep(3)
        print('done!')

    # Load train and test separately
    df = pd.read_csv(os.path.join(ROOT_DIR, DATA_DIR, 'USPS', 'train.gz'), header=None)
    X_train = []
    y_train = []
    for i in range(df.shape[0]):
        temp = df[0][i].split()
        X_train.append(np.array(list(map(float, temp[1:]))).reshape(16, 16))
        y_train.append(int(float(temp[0])))
    X_train = np.array(X_train)
    y_train = np.array(y_train)

    df = pd.read_csv(os.path.join(ROOT_DIR, DATA_DIR, 'USPS', 'test.gz'), header=None)
    X_test = []
    y_test = []
    for i in range(df.shape[0]):
        temp = df[0][i].split()
        X_test.append(np.array(list(map(float, temp[1:]))).reshape(16, 16))
        y_test.append(int(float(temp[0])))
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    # Eventually resize
    if resize is not None:
        assert type(resize) is tuple
        X_train = np.array([cv2.resize(X_train[i], dsize=resize) for i in range(X_train.shape[0])])
        X_test = np.array([cv2.resize(X_test[i], dsize=resize) for i in range(X_test.shape[0])])

    return (X_train, y_train), (X_test, y_test)


if __name__ == '__main__':
    x, _ = load_USPS()
    print(x[0])
