from constants import RANDOM_SEED, ROOT_DIR, BATCH_SIZE, DATA_DIR
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from downloads import load_CIFAR10, load_CIFAR100, load_MNIST, load_USPS, load_dataset
import cv2
from autoencoder import *
import time
import copy


class WrapperDataset(Dataset):
    """
    A Dataset object wrapper for the initialized from numpy arrays.
    """

    def __init__(self, X, y):
        # If channel is not provided assume it is only one channel
        if len(X.shape) < 4:
            X = np.expand_dims(X, axis=-1)

        # Shape has to be (Batch size, num_channels, width, height)
        self.X = np.moveaxis(X, source=-1, destination=1).astype(np.float32)
        # To return LongTensor
        self.y = y.astype(np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        """
        Retrieve images by their index.
        :param idx: index of the image in the dataset.
        :return: (X[idx], y[idx]): the image and its label
        """
        return self.X[idx], self.y[idx]

    def get_X(self):
        return self.X

    def get_y(self):
        return self.y


def get_data_loader(X, y, batch_size=BATCH_SIZE, shuffle=True, num_workers=0):
    """
    Generate a DataLoader object given a two numpy arrays.
    :param X: numpy array of samples
    :param y: numpy array of labels
    :param shuffle: if True, shuffle samples
    :param num_workers:
    :return: a DataLoader object initialized with X and y
    """
    y = y.flatten()  # Flatten for the DataLoader
    dataset = WrapperDataset(X=X, y=y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def add_noise_dataset(X, y, mean=3.0, std=0.001, classes_to_distort=None):
    """
    Add gaussian noise images of the dataset.
    :param X: numpy array of samples
    :param y: numpy array of labels
    :param mean: mean of the distribution
    :param std: standard deviation
    :param classes_to_distort: list of labels
    :return: X_noisy: a noisy version of X
    """
    X_noisy = np.empty(X.shape)

    for i in tqdm(range(X.shape[0]), desc='Adding Gaussian noise'):
        if classes_to_distort is None:
            X_noisy[i] = gaussian_noise(image=X[i], mean=mean, sigma=std)
        else:
            if y[i] in classes_to_distort:
                X_noisy[i] = gaussian_noise(image=X[i], mean=mean, sigma=std)
            else:
                X_noisy[i] = X[i].copy()

    return X_noisy


def channel_shift_dataset(X, y, channel=1, shift=5, classes_to_distort=None):
    """
    Add gaussian noise images of the dataset.
    :param X: numpy array of samples
    :param y: numpy array of labels
    :param channel: channel to shift
    :param shift: shift amount to apply
    :param classes_to_distort: list of labels
    :return: X_noisy: a noisy version of X
    """
    X_noisy = np.empty(X.shape)
    channels_left = [0, 1, 2]
    channels_left.remove(channel)

    for i in tqdm(range(X.shape[0]), desc='Shifting '):
        if classes_to_distort is None or y[i] in classes_to_distort:
            X_noisy[i, :, :, channel] = X[i, :, :, channel].copy() + shift
            for cl in channels_left:
                X_noisy[i, :, :, cl] = X[i, :, :, cl].copy()
        else:
            X_noisy[i] = X[i].copy()
        X_noisy[i] = np.clip(X_noisy[i], 0, 255)
    return X_noisy


def blur_dataset(X, y, std_dev=0.5, classes_to_distort=None):
    """
    Blur images of the dataset.
    :param X: numpy array of samples
    :param y: numpy array of labels
    :param std_dev: standard deviation of the blur
    :param classes_to_distort: list of labels
    :return: X_noisy: a blurred version of X
    """
    X_noisy = np.empty(X.shape)

    # Paper: "For both datasets, the size of the blur kernel is set to 4 times the blur standard deviation sigma."
    kernel_size = int(4 * std_dev)

    # sigmaY = Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set to be equal to sigmaX
    for i in range(X.shape[0]):
        if classes_to_distort is None:
            X_noisy[i] = cv2.GaussianBlur(X[i], ksize=(kernel_size, kernel_size), sigmaX=std_dev, sigmaY=0)
        else:
            if y[i] in classes_to_distort:
                X_noisy[i] = cv2.GaussianBlur(X[i], ksize=(kernel_size, kernel_size), sigmaX=std_dev, sigmaY=0)
            else:
                X_noisy[i] = X[i].copy()

    return X_noisy


def gaussian_noise(image, mean=0., sigma=0.1):
    """
    Add gaussian noise to an input image
    :param image: image to be distorted
    :param mean: mean of the distribution
    :param sigma: standard deviation
    :return: noisy: the noisy image
    """
    np.random.seed(RANDOM_SEED)
    if image.ndim == 3:
        row, col, ch = image.shape
        gauss = np.random.normal(mean, sigma, (row, col, ch))
    else:
        row, col = image.shape
        gauss = np.random.normal(mean, sigma, (row, col))
    noisy = (image + gauss)
    noisy = noisy.clip(0, 255)
    return noisy


def gen_distorted_dataset(dataset, distortion_type, mean, std, classes_to_distort=None):
    """
    Generate and save dataset with given distortion type (numpy array with dimension (w, h, c))
    :param dataset: ['CIFAR_10', 'CIFAR_100', 'MNIST', 'USPS']
    :param distortion_type: blur or additive gaussian white noise
    :type distortion_type: ['blur', 'AWGN', 'red_shift', 'green_shift', 'blue_shift]
    :param std: standard deviation of the distribution
    :param mean: if distortion_type = 'AWGN', the mean of the Gaussian Distribution
    :param classes_to_distort: distort only samples that belong to the list of classes provided
    :type: classes: None or list
    """

    (X_train, y_train), (X_test, y_test) = load_dataset(dataset)

    if distortion_type == 'AWGN':
        X_train_noisy = add_noise_dataset(X_train, y_train, mean=mean, std=std, classes_to_distort=classes_to_distort)
        X_test_noisy = add_noise_dataset(X_test, y_test, mean=mean, std=std, classes_to_distort=classes_to_distort)
    elif distortion_type == 'blur':
        X_train_noisy = blur_dataset(X_train, y_train, std_dev=std, classes_to_distort=classes_to_distort)
        X_test_noisy = blur_dataset(X_test, y_test, std_dev=std, classes_to_distort=classes_to_distort)
    elif distortion_type in ['red_shift', 'green_shift', 'blue_shift']:
        colors_channels = {'red_shift': 0,
                           'green_shift': 1,
                           'blue_shift': 2}
        channel = colors_channels[distortion_type]
        X_train_noisy = channel_shift_dataset(X_train, y_train, channel=channel, shift=mean,
                                              classes_to_distort=classes_to_distort)
        X_test_noisy = channel_shift_dataset(X_test, y_test, channel=channel, shift=mean,
                                             classes_to_distort=classes_to_distort)
    elif distortion_type == 'AE_shift':
        autoencoder = CIFAR10Autoencoder()
        path = os.path.join(ROOT_DIR, MODELS_DIR, 'autoencoders', 'CIFAR_10_autoencoder.pt')
        autoencoder.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
        X_train_noisy = shift_AE_dataset(dataset, X_train, y_train, autoencoder=autoencoder, shift=mean,
                                         classes_to_distort=classes_to_distort)
        X_test_noisy = shift_AE_dataset(dataset, X_test, y_test, autoencoder=autoencoder, shift=mean,
                                        classes_to_distort=classes_to_distort)
    else:
        raise RuntimeError("Distortion type must be 'blur', 'AWGN', 'AE_shift")

    save_path = os.path.join(ROOT_DIR, DATA_DIR, dataset, distortion_type + '-m=' + str(mean) + '-std=' + str(std))

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Saving files in data folder
    np.save(os.path.join(save_path, "X_train.npy"), X_train)
    np.save(os.path.join(save_path, "X_test.npy"), X_test)
    np.save(os.path.join(save_path, "X_train_noisy.npy"), X_train_noisy)
    np.save(os.path.join(save_path, "X_test_noisy.npy"), X_test_noisy)
    np.save(os.path.join(save_path, "y_train.npy"), y_train)
    np.save(os.path.join(save_path, "y_test.npy"), y_test)
    time.sleep(3)


def shift_AE_dataset(dataset, X, y, autoencoder, shift, classes_to_distort=None, return_encoding=False):
    """
    Generate a shifted version of the dataset using an autoencoder to shift the features in the latent space.
    :param dataset: dataset
    :param autoencoder:
    :param shift:
    :param classes_to_distort:
    :param return_encoding:
    :return:
    """
    X, _, _ = image_preprocessing(X, scale_only=False)
    X_shifted = []
    encodings_shifted = []

    data_loader = get_data_loader(X, y, shuffle=False)

    # Train
    for i, data in enumerate(data_loader):
        inputs, labels = data

        encoded, outputs = autoencoder(inputs)
        mask = torch.ByteTensor(np.isin(labels.numpy(), classes_to_distort))
        if classes_to_distort is None:
            encoded_shifted = torch.where(mask.view(inputs.shape[0], 1, 1, 1), encoded, encoded + shift)
        else:
            encoded_shifted = torch.where(mask.view(inputs.shape[0], 1, 1, 1), encoded + shift, encoded)

        for encoding in encoded_shifted:
            encodings_shifted.append(encoding.detach().numpy().flatten())
        outputs_shifted = autoencoder.decoder(encoded_shifted)
        for output in outputs_shifted:
            X_shifted.append(output.detach().numpy())

    X_shifted = np.array(X_shifted)
    X_shifted = np.moveaxis(X_shifted, 1, 3)

    if return_encoding:
        return X_shifted, np.array(encodings_shifted)
    else:
        return X_shifted


def image_preprocessing(X, seq_mean=None, seq_std=None, scale_only=False):
    """
    Normalize images to have each pixel value in a range [0, 1].
    :param X: numpy array of images with the following format: (samples, width, height, [channels]).
              If the tuple has three dimensions: (samples, width, height), channels is inferred = 1.
    :param scale_only: if True, just scale the values, otherwise normalize them
    :return: X: normalized images
    """

    # Check if channels > 1
    if X.ndim > 3:
        channels = X.shape[-1]
    else:
        channels = 1

    # Cast from int to float
    X = X.astype(float)

    if scale_only:
        return X / 255.0

    else:
        # Compute mean and std for each channel
        if seq_mean is None and seq_std is None:
            seq_mean = np.mean(X, axis=tuple(range(3)))
            seq_std = np.std(X, axis=tuple(range(3)))

        if channels == 1:
            X[:, :, :] = (X[:, :, :] - seq_mean) / seq_std
        else:
            for ch in range(channels):
                X[:, :, :, ch] = (X[:, :, :, ch] - seq_mean[ch]) / seq_std[ch]

        return X, seq_mean, seq_std
