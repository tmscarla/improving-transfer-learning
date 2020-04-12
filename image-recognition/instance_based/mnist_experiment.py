import numpy as np
from training import *
from instance_based.models import InstanceMNISTNet
from downloads import load_MNIST, load_EMNIST
from utils import *
from instance_based.instance_selection import *
import sys
from models import FFSimpleNet
from constants import device


class MNIST_Experiment():

    def _generate_data(self, dataset, batch_size=128):
        if dataset == 'EMNIST':
            (X_train, y_train), (X_test, y_test) = load_EMNIST(split='letters')
            y_train, y_test = y_train-1, y_test-1 # Adjust labels from [1-26] to [0-25]
        elif dataset == 'MNIST':
            (X_train, y_train), (X_test, y_test) = load_MNIST()
        else:
            raise Exception('Dataset not supported')

        # Split and scale
        (X_train, y_train), (X_valid, y_valid) = dataset_split(X_train, y_train, return_data='samples')

        X_train_sc, X_mean, X_std = image_preprocessing(X_train, scale_only=False)
        X_valid_sc, _, _ = image_preprocessing(X_valid, seq_mean=X_mean, seq_std=X_std, scale_only=False)
        X_test_sc, _, _ = image_preprocessing(X_test, seq_mean=X_mean, seq_std=X_std, scale_only=False)

        # Dataloaders
        train_dl = get_data_loader(X_train_sc, y_train, batch_size=batch_size)
        valid_dl = get_data_loader(X_valid_sc, y_valid, batch_size=batch_size)
        test_dl = get_data_loader(X_test_sc, y_test, batch_size=batch_size)

        return (X_train_sc, X_valid_sc, X_test_sc), (y_train, y_valid, y_test), (train_dl, valid_dl, test_dl)

    def training_model(self, model, dataset, split='letters', batch_size=128, epochs=5):

        _, _, (train_dl, valid_dl, test_dl) = self._generate_data(dataset)

        print('Training from scratch on {}...'.format(dataset))

        # Training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        train(model, train_dl, valid_dl, test_dl, optimizer=optimizer, device=device, criterion=criterion,
              early_stopping=False, epochs=epochs, flatten=True)
        return model

    def instance_based(self, flatten=True):

        # Train first on EMNIST
        if flatten:
            model_instance_based = FFSimpleNet(output_dim=26)
        else:
            model_instance_based = InstanceMNISTNet(output_dim=26)
        self.training_model(model=model_instance_based, dataset='EMNIST', epochs=1)

        # Load MNIST and change the last layer of the network
        (X_train_sc_M, X_valid_sc_M, X_test_sc_M), (y_train_M, y_valid_M, y_test_M), \
            (train_dl_M, valid_dl_M, test_dl_M) = self._generate_data('MNIST')
        model_instance_based.fc2 = nn.Linear(model_instance_based.fc2.in_features, 10)

        # Select indices
        optimizer = torch.optim.Adam(model_instance_based.parameters(), lr=0.01)
        criterion = torch.nn.CrossEntropyLoss()

        selected_indices = instance_selection_no_hessian(model=model_instance_based, X_train=X_train_sc_M,
                                              y_train=y_train_M,
                                              X_valid=X_valid_sc_M, y_valid=y_valid_M,
                                              criterion=criterion, flatten=flatten)
        X_train_M_sel, y_train_M_sel = X_train_sc_M[selected_indices], y_train_M[selected_indices]
        train_dl_M_sel = get_data_loader(X_train_M_sel, y_train_M_sel)

        train(model_instance_based, train_dl_M_sel, valid_dl_M, test_dl_M, optimizer=optimizer, device=device,
              criterion=criterion, early_stopping=False, epochs=20)


if __name__ == '__main__':
    experiment = MNIST_Experiment()

    ### EMNIST Training ###
    experiment.instance_based()
