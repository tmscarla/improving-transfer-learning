import numpy as np
import torch
import torch.nn as nn
from constants import *
import sys
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms


class CIFAR10Autoencoder(nn.Module):
    def __init__(self):
        super(CIFAR10Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=12, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=48, out_channels=24, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=24, out_channels=12, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=12, out_channels=3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded


def train_autoencoder(autoencoder, train_loader, val_loader, optimizer, criterion, device,
                      writer, start_epoch=None, scheduler=None, epochs=100, early_stopping=True,
                      model_path=ROOT_DIR + '/models', flatten=False, pbar=True):
    autoencoder.train()
    best_val_loss = sys.float_info.max  # Highest possible value

    # Store losses and accuracies for plotting
    train_losses = []
    val_losses = []

    # Initialization
    epoch = 0
    bad_epochs = 0
    end = False
    if start_epoch is not None:
        epoch = start_epoch
    if val_loader is None:
        early_stopping = False

    # Training until number of epochs is reached or best validation loss is not overcome after PATIENCE epochs
    while not end:
        if epoch == epochs and not early_stopping:
            break

        # Initialization for current epoch
        autoencoder.train()
        autoencoder.to(device)
        epoch_start_time = time.time()
        losses_dict = dict()
        running_loss = 0.0

        # Display progress bar or a string after each epoch
        if pbar:
            tqdm_train = tqdm(train_loader, desc='Epoch {:4d}'.format(epoch),
                              bar_format='{desc}: {percentage:3.0f}%|{bar}|'
                                         ' {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]')
        else:
            tqdm_train = train_loader

        for i, data in enumerate(tqdm_train):
            inputs, _ = data
            # Flatten inputs if model is a FF network
            if flatten:
                inputs = inputs.view(inputs.shape[0], -1)
            # Move to GPU
            inputs = inputs.to(device)
            # Forward pass
            encoded, outputs = autoencoder(inputs)
            # Loss computation
            loss = criterion(outputs, inputs)
            # Reset gradient
            optimizer.zero_grad()
            # Gradient computation
            loss.backward()
            # Optimization step
            optimizer.step()

            running_loss += loss.item()

            # Last iteration
            if i == len(train_loader) - 1:
                # Scheduler
                if scheduler is not None:
                    scheduler.step()

                # Train
                train_loss = running_loss / len(train_loader)
                train_losses.append(train_loss)

                # Validation
                if val_loader is not None:
                    val_loss = validate_autoencoder(autoencoder, val_loader, criterion, device, flatten)
                    val_losses.append(val_loss)
                else:
                    val_loss = 0.0

                # Logging
                if writer is not None:
                    writer.add_scalar('Loss/train', train_loss, epoch)
                    writer.add_scalar('Loss/val', val_loss, epoch)

                # Progress bar
                losses_dict['Train loss'] = '{:.6f}'.format(train_loss)
                losses_dict['Valid loss'] = '{:.6f}'.format(val_loss)
                if pbar:
                    tqdm_train.set_postfix(losses_dict, refresh=True)
                else:
                    print('Epoch: {:4d} | Time: {:5.2f}s | Train loss: {:.6f}'
                          ' | Valid loss: {:.6f} '.format(epoch,
                                                          (time.time() - epoch_start_time),
                                                          train_loss, val_loss))

                # Store model that has smallest validation loss
                if val_loss < best_val_loss:
                    print('Best model found!')
                    path = os.path.join(ROOT_DIR, MODELS_DIR, model_path)
                    torch.save(autoencoder.state_dict(), path)
                    best_val_loss = val_loss
                    bad_epochs = 0
                else:
                    bad_epochs += 1

                if early_stopping and bad_epochs == PATIENCE:
                    end = True
        epoch += 1


def validate_autoencoder(model, val_loader, criterion, device, flatten=False):
    """
    Validate a neural network model through a validation set.
    :param model: a neural network model, which inherits from nn.Module
    :param val_loader: DataLoader object with validation data
    :param criterion: loss function
    :param device: physical device on which the model will be validated
    :param flatten: flatten input if the model is a FF neural network
    :return: (epoch_loss, epoch_acc): loss and accuracy of the model
    """

    model.eval()
    model.to(device)
    running_loss = 0.0

    with torch.no_grad():
        for data in val_loader:
            inputs, _ = data

            if flatten:
                inputs = inputs.view(inputs.shape[0], -1)

            # Move to GPU
            inputs = inputs.to(device)
            # Forward pass
            _, outputs = model(inputs)

            # Loss function computation
            loss = criterion(outputs, inputs)

            running_loss += loss.item()

    epoch_loss = running_loss / len(val_loader)

    return epoch_loss


def visualize_AE_samples(inputs, labels, outputs, outputs_shifted, shift, limit=None):
    """
    Visualize a batch in a comprehensive way. It displays in a single figure the input image,
    the clean output of the autoencoder and the shifted output.
    :param inputs: input images
    :param labels: labels of the samples
    :param outputs: output images not shifted
    :param outputs_shifted: images decoded from the shifted latent representation
    :param shift: shift amount
    :param limit: limit the number of visualized samples. If None,
    :return:
    """
    if limit is None:
        limit = len(inputs)

    for idx in range(len(inputs)):
        if idx == limit: break

        plt.figure()

        # Display input
        plt.subplot(1, 3, 1)
        plt.title("Input | Label = {}".format(CIFAR_10_LABELS[labels[idx].item()]))
        input_img = inputs[idx].detach().numpy().transpose((1, 2, 0))
        plt.imshow(input_img)

        plt.subplot(1, 3, 2)
        plt.title("Output")
        output_img = outputs[idx].detach().numpy().transpose((1, 2, 0))
        plt.imshow(output_img)
        plt.show()

        plt.subplot(1, 3, 3)
        plt.title("Output shifted | Shift = {}".format(shift))
        output_shifted_img = outputs_shifted[idx].detach().numpy().transpose((1, 2, 0))
        plt.imshow(output_shifted_img)
        plt.show()
