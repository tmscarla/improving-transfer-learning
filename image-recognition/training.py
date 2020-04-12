import os
import time

import numpy
import torch
from sklearn.model_selection import StratifiedShuffleSplit
# from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

from constants import *
import sys
from torch import nn

from constants import RANDOM_SEED, ROOT_DIR, DATA_DIR, VALIDATION_PERCENTAGE
from datasets import *
from tqdm import tqdm

from datasets import image_preprocessing, get_data_loader
from downloads import load_CIFAR10, load_CIFAR100, load_MNIST, load_USPS
from models import *
from nets.squeezenet import squeezenet
from utils import save_log, gen_plots, select_classes, convert_labels
from boosting.weighted_loss import init_loss


def train(model, train_loader, val_loader, test_loader, optimizer, criterion, device,
          writer=None, start_epoch=None, scheduler=None, epochs=100, early_stopping=True,
          save_model=False, model_path=ROOT_DIR + '/models', flatten=False, pbar=True,
          verbose=True):
    """
    Wrapper function to train a neural network model.
    :param model: a neural network model, which inherits from nn.Module
    :param train_loader: DataLoader object with training data
    :param val_loader: DataLoader object with validation data
    :param test_loader: DataLoader object with test data
    :type test_loader: DataLoader or None
    :param optimizer: optimizer to adjust weights during training
    :param criterion: loss function
    :param device: physical device on which the model will be trained
    :param writer: optional tensorboard SummaryWriter to record accuracies and losses
    :param start_epoch: start counting epochs from start_epoch value on
    :type start_epoch: int
    :param scheduler: optional scheduler to dynamically adjust the learning rate
    :param epochs: number of epochs to train the model. If early_stopping is True this parameter is ignored.
    :param early_stopping: stop training after a number of epochs with no improvements on the validation set == PATIENCE
    :param save_model: save the best model on disk
    :param model_path: path to save weights of the best model
    :param flatten: flatten inputs if model is a FF network
    :param pbar: if True, display a progress bar, otherwise display a line after each epoch
    :param verbose: if False, suppress every output to the stdout
    """
    # Start training
    model.train()
    best_val_loss = sys.float_info.max  # Highest possible value

    # Store losses and accuracies for plotting
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []

    # Initialization
    epoch = 0
    bad_epochs = 0
    end = False

    # Check if binary prediction labels are [-1, 1] or [0, 1]
    tanh = False
    unique_labels = np.unique(val_loader.dataset.y)
    if len(unique_labels) == 2 and -1 in unique_labels:
        tanh = True

    # Adjust epochs count
    if start_epoch is not None:
        epoch = start_epoch
        epochs = start_epoch + epochs
    if val_loader is None:
        early_stopping = False

    # Training until number of epochs is reached or best validation loss is not overcome after PATIENCE epochs
    while not end:
        if epoch == epochs and not early_stopping:
            break

        # Initialization for current epoch
        model.train()
        model.to(device)
        epoch_start_time = time.time()
        losses_dict = dict()
        running_loss = 0
        running_corrects = 0
        total = 0

        # Display progress bar or a string after each epoch
        if pbar:
            tqdm_train = tqdm(train_loader, desc='Epoch {:4d}'.format(epoch),
                              bar_format='{desc}: {percentage:3.0f}%|{bar}|'
                                         ' {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]',
                              disable=not verbose)
        else:
            tqdm_train = train_loader

        # Actual training
        for i, data in enumerate(tqdm_train):
            inputs, labels = data

            # Flatten inputs if model is a FF network
            if flatten:
                inputs = inputs.view(inputs.shape[0], -1)

            # Move to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            # Reset gradient
            optimizer.zero_grad()
            # Forward pass
            outputs = model(inputs)

            # Binary prediction
            if outputs.shape[1] is 1:
                outputs = outputs.cpu()

                if tanh:
                    predicted = torch.where(outputs.data > 0, torch.ones(outputs.data.shape),
                                            torch.ones(outputs.data.shape) * (-1))
                else:
                    predicted = torch.where(outputs.data > 0.5, torch.ones(outputs.data.shape),
                                            torch.zeros(outputs.data.shape))
                labels = labels.float()
                outputs.view(-1)

            # Multi-class prediction
            else:
                _, predicted = torch.max(outputs.data, 1)
                labels = labels.long()

            predicted = predicted.view(-1)
            predicted = predicted.to(device)
            # Loss function computation
            loss = criterion(outputs, labels)
            # Gradient computation
            loss.backward()
            # Optimization step
            optimizer.step()

            running_loss += loss.item()
            running_corrects += (predicted == labels).sum().item()
            total += labels.size(0)

            # Last iteration
            if i == len(train_loader) - 1:
                # Scheduler
                if scheduler is not None:
                    scheduler.step()

                # Train
                # train_loss = running_loss / len(train_loader.dataset.X)
                train_loss = running_loss
                train_acc = running_corrects / total
                train_losses.append(train_loss)
                train_accuracies.append(train_acc)

                # Validation
                if val_loader is not None:
                    val_loss, val_acc = validate(model, val_loader, criterion, device, flatten)
                    val_losses.append(val_loss)
                    val_accuracies.append(val_acc)
                else:
                    val_loss = 0
                    val_acc = 0

                # Test
                if test_loader is not None:
                    test_acc = evaluate(model, test_loader, device, flatten)
                    test_accuracies.append(test_acc)
                else:
                    test_acc = 0

                # Logging
                if writer is not None:
                    writer.add_scalar('Loss/train', train_loss, epoch)
                    writer.add_scalar('Accuracy/train', train_acc, epoch)
                    writer.add_scalar('Loss/val', val_loss, epoch)
                    writer.add_scalar('Accuracy/val', val_acc, epoch)
                    writer.add_scalar('Accuracy/test', test_acc, epoch)

                # Progress bar
                losses_dict['Train loss'] = '{:.3f}'.format(train_loss)
                losses_dict['Train acc'] = '{:.2f}%'.format(train_acc * 100)
                losses_dict['Valid loss'] = '{:.3f}'.format(val_loss)
                losses_dict['Valid acc'] = '{:.2f}%'.format(val_acc * 100)
                if verbose:
                    if pbar:
                        tqdm_train.set_postfix(losses_dict, refresh=True)
                    else:
                        print('Epoch: {:4d} | Time: {:5.2f}s | Train loss: {:.5f} | Train acc: {:.3f}'
                              ' | Valid loss: {:.5f} | Valid acc: {:.3f}'.format(epoch,
                                                                                 (time.time() - epoch_start_time),
                                                                                 train_loss, train_acc, val_loss,
                                                                                 val_acc))

                # Store model that has smallest validation loss
                if val_loss < best_val_loss:
                    path = os.path.join(ROOT_DIR, MODELS_DIR, model_path)
                    if save_model:
                        torch.save(model.state_dict(), path)
                    best_val_loss = val_loss
                    bad_epochs = 0
                    best_model = copy.deepcopy(model)
                else:
                    bad_epochs += 1

                if early_stopping and bad_epochs == PATIENCE:
                    end = True
                    model = best_model
                    break
                epoch += 1

    return train_losses, train_accuracies, val_accuracies, val_losses, epoch, model


def validate(model, val_loader, criterion, device, flatten=False):
    """
    Validate a neural network model through a validation set.
    :param model: a neural network model, which inherits from nn.Module
    :param val_loader: DataLoader object with validation data
    :param criterion: loss function
    :param device: physical device on which the model will be validated
    :param flatten: flatten input if the model is a FF neural network
    :return: (epoch_loss, epoch_acc): loss and accuracy of the model
    """

    # Setup
    model.eval()
    model.to(device)

    running_loss = 0.0
    running_corrects = 0
    total = 0

    # Check if binary prediction labels are [-1, 1] or [0, 1]
    tanh = False
    unique_labels = np.unique(val_loader.dataset.y)
    if len(unique_labels) == 2 and -1 in unique_labels:
        tanh = True

    with torch.no_grad():
        for data in val_loader:
            inputs, labels = data

            if flatten:
                inputs = inputs.view(inputs.shape[0], -1)

            # Move to GPU
            inputs, labels = inputs.to(device), labels.to(device)
            # Forward pass
            outputs = model(inputs)

            # Binary prediction
            if outputs.shape[1] is 1:
                outputs = outputs.cpu()

                if tanh:
                    predicted = torch.where(outputs.data > 0, torch.ones(outputs.data.shape),
                                            torch.ones(outputs.data.shape) * (-1))
                else:
                    predicted = torch.where(outputs.data > 0.5, torch.ones(outputs.data.shape),
                                            torch.zeros(outputs.data.shape))
                labels = labels.float()
                outputs.view(-1)
            # Multi-class prediction
            else:
                _, predicted = torch.max(outputs.data, 1)

            predicted = predicted.view(-1)
            predicted = predicted.to(device)

            # Loss function computation
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_corrects += (predicted == labels).sum().item()
            total += labels.size(0)

    # epoch_loss = running_loss / len(val_loader)
    epoch_loss = running_loss
    epochs_acc = running_corrects / total

    return epoch_loss, epochs_acc


def evaluate(model, test_loader, device, flatten=False):
    """
    Evaluate a neural network model through a test set.
    :param model: a neural network model, which inherits from nn.Module
    :param test_loader: DataLoader object with test data
    :param device: physical device on which the model will be evaluated
    :param flatten: flatten input if the model is a FF neural network
    :return: accuracy: accuracy of the model on test data
    """
    model.eval()
    model.to(device)

    correct = 0
    total = 0

    # Check if binary prediction labels are [-1, 1] or [0, 1]
    tanh = False
    unique_labels = np.unique(test_loader.dataset.y)
    if len(unique_labels) == 2 and -1 in unique_labels:
        tanh = True

    with torch.no_grad():
        for data in test_loader:
            # Unwrap tuple
            images, labels = data

            if flatten:
                images = images.view(images.shape[0], -1)

            # Move to GPU
            images, labels = images.to(device), labels.to(device)
            # Forward pass
            outputs = model(images)

            # Binary prediction
            if outputs.shape[1] is 1:
                outputs = outputs.cpu()

                if tanh:
                    predicted = torch.where(outputs.data > 0, torch.ones(outputs.data.shape),
                                            torch.ones(outputs.data.shape) * (-1))
                else:
                    predicted = torch.where(outputs.data > 0.5, torch.ones(outputs.data.shape),
                                            torch.zeros(outputs.data.shape))
                labels = labels.float()
                outputs.view(-1)
            # Multi-class prediction
            else:
                _, predicted = torch.max(outputs.data, 1)

            predicted = predicted.view(-1)
            predicted = predicted.to(device)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Return accuracy
    accuracy = correct / total
    return accuracy


def compute_features(model, data_loader, device, flatten=False):
    # Actual training
    features = []
    for i, data in enumerate(data_loader):
        inputs, labels = data

        # Flatten inputs if model is a FF network
        if flatten:
            inputs = inputs.view(inputs.shape[0], -1)

        # Move to GPU
        inputs, labels = inputs.to(device), labels.to(device)

        # Compute features for each sample
        for single_input in inputs:
            single_input = single_input.view(1, *single_input.shape)
            single_features = model.forward_features(single_input).cpu().detach().numpy()
            features.append(single_features[0])

    return np.array(features)


def train_baseline(dataset, model_name, noisy=False, distortion_type='AWGN', distortion_amount=25,
                   flatten=False, verbose=True, classes=None):
    """
    Train a baseline model using a specific dataset.
    :param dataset: dataset to train the model on
    :param model_name: name of the neural network model
    :param noisy: if True, train the model on
    :param distortion_type: either 'blur' or 'AWGN'
    :param distortion_amount: severity of the distortion
    :param flatten: flatten the input image to use it in a FF network model
    :param verbose: add verbosity
    :return:
    """
    assert dataset in DSETS

    # Set seeds
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    # Set model path
    model_path = os.path.join('baselines', model_name + '.pt')

    # Train baseline on noisy data
    if noisy:
        (X_train, y_train), (X_test, y_test) = (np.load(os.path.join(ROOT_DIR, DATA_DIR, dataset, distortion_type +
                                                                     '-' + str(distortion_amount),
                                                                     'X_train_noisy.npy')),
                                                np.load(os.path.join(ROOT_DIR, DATA_DIR, dataset,
                                                                     distortion_type + '-' + str(distortion_amount),
                                                                     'y_train.npy'))), \
                                               (np.load(os.path.join(ROOT_DIR, DATA_DIR, dataset,
                                                                     distortion_type + '-' + str(distortion_amount),
                                                                     'X_test_noisy.npy')),
                                                np.load(os.path.join(ROOT_DIR, DATA_DIR, dataset,
                                                                     distortion_type + '-' + str(distortion_amount),
                                                                     'y_test.npy')))

    # Train baseline on clean data
    else:
        if dataset == 'CIFAR_10':
            (X_train, y_train), (X_test, y_test) = load_CIFAR10()
            if classes is not None:
                X_train, y_train = select_classes(X_train, y_train, classes, convert_labels=True)
                X_test, y_test = select_classes(X_test, y_test, classes, convert_labels=True)

            if len(np.unique(y_train)) > 2:
                baseline_net = SimpleBaselineNet(output_dim=len(classes))
            else:
                if model_name == 'SimpleBaselineBinaryNetTanh':
                    y_train = convert_labels(y_train, [0, 1], [-1, 1])
                    y_test = convert_labels(y_test, [0, 1], [-1, 1])
                    baseline_net = SimpleBaselineBinaryNet(activation='tanh')
                elif model_name == 'SimpleBaselineBinaryNet':
                    baseline_net = SimpleBaselineBinaryNet(activation='sigmoid')
                elif model_name == 'SimplerBaselineBinaryNetTanh':
                    y_train = convert_labels(y_train, [0, 1], [-1, 1])
                    y_test = convert_labels(y_test, [0, 1], [-1, 1])
                    baseline_net = SimpleBaselineBinaryNet(activation='tanh', num_conv=32, num_ff=32)
        elif dataset == 'CIFAR_100':
            (X_train, y_train), (X_test, y_test) = load_CIFAR100()
            if model_name == 'SqueezeNetBaseline':
                baseline_net = squeezenet()
            else:
                baseline_net = ACNBaselineNet()

        elif dataset == 'MNIST':
            (X_train, y_train), (X_test, y_test) = load_MNIST()
            if classes is not None:
                X_train, y_train = select_classes(X_train, y_train, classes, convert_labels=True)
                X_test, y_test = select_classes(X_test, y_test, classes, convert_labels=True)

            if len(np.unique(y_train)) > 2:
                baseline_net = FFSimpleNet()
            else:
                baseline_net = FFBinaryNet()
            flatten = True

        elif dataset == 'USPS':
            (X_train, y_train), (X_test, y_test) = load_USPS(resize=(28, 28))
            baseline_net = FFSimpleNet()
            flatten = True

        else:
            raise RuntimeError("Dataset not in the predefined list: {}".format(DSETS))

    # Scale pixels values
    X_train, X_mean, X_std = image_preprocessing(X_train, scale_only=False)
    X_test, _, _ = image_preprocessing(X_test, seq_mean=X_mean, seq_std=X_std, scale_only=False)

    # Flatten for the dataloader
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Stratified split of training and validation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_PERCENTAGE, random_state=RANDOM_SEED)
    train_idx, val_idx = next(sss.split(X_train, y_train))
    (X_train, X_valid) = X_train[train_idx], X_train[val_idx]
    (y_train, y_valid) = y_train[train_idx], y_train[val_idx]

    # Generating data loaders
    train_loader_clean = get_data_loader(X_train, y_train)
    val_loader_clean = get_data_loader(X_valid, y_valid, shuffle=False)
    test_loader_clean = get_data_loader(X_test, y_test, shuffle=False)

    # Logger
    if noisy:
        writer = SummaryWriter('runs/' + dataset + '_baseline_noisy')
    else:
        writer = SummaryWriter('runs/' + dataset + '_baseline_clean')

    # Optimizer and criterion
    optimizer = torch.optim.Adam(baseline_net.parameters())
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 60, gamma=0.02, last_epoch=-1)

    if len(np.unique(y_train)) > 2:
        criterion = nn.CrossEntropyLoss()
    elif len(np.unique(y_train)) == 2 and -1 in np.unique(y_train):
        criterion, _ = init_loss(X_train, loss='exp')
    else:
        criterion = nn.BCELoss()
    baseline_net.to(device)

    # Training and evaluation
    if verbose:
        print('Starting {} baseline training on {}'.format(baseline_net.__class__.__name__, dataset))

    train(model=baseline_net, train_loader=train_loader_clean, val_loader=val_loader_clean,
          test_loader=test_loader_clean, optimizer=optimizer, criterion=criterion, device=device, model_path=model_path,
          writer=writer, save_model=True, scheduler=None, flatten=flatten, early_stopping=True)
    acc = evaluate(baseline_net, test_loader_clean, device, flatten)

    if verbose:
        print('Your baseline accuracy on ' + dataset + ' (x_test_clean) = %.3f' % acc)
