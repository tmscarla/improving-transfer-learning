import torch
from sklearn.datasets import make_classification
import toy_utils
from sklearn.preprocessing import StandardScaler
from toy_models import *
from instance_based.instance_selection import instance_selection, instance_selection_no_hessian
import toy_constants
import matplotlib.pyplot as plt
import copy
from constants import *
import numpy as np
from training import train
import datetime


class ToyInstanceSelection():
    def __init__(self, n_samples, input_dim, hidden_dim, output_dim, theta):
        self.n_samples = n_samples
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.theta = theta
        self.setup_clean()
        self.setup_noisy()

    def setup_clean(self):
        """
        Setup a clean setting a train a model on it. Numpy arrays of samples such as X_train, X_test
        and X_valid are not scaled. Dataloaders object instead are initialized with scaled arrays.
        """
        # Generate points
        self.X, self.y = make_classification(n_samples=self.n_samples, n_features=self.input_dim,
                                             n_informative=self.input_dim,
                                             n_redundant=0, n_classes=self.output_dim, n_clusters_per_class=1)

        # Train-Test and Train-Validation
        self.train_idx_, self.test_idx = toy_utils.dataset_split(
            self.X, self.y, perc=toy_constants.TEST_PERCENTAGE, return_data='indices')
        self.X_train, self.X_test = self.X[self.train_idx_], self.X[self.test_idx]
        self.y_train, self.y_test = self.y[self.train_idx_], self.y[self.test_idx]

        self.train_idx, self.valid_idx = toy_utils.dataset_split(
            self.X_train, self.y_train, perc=toy_constants.VALIDATION_PERCENTAGE, return_data='indices')
        self.X_train, self.X_valid = self.X_train[self.train_idx], self.X_train[self.valid_idx]
        self.y_train, self.y_valid = self.y_train[self.train_idx], self.y_train[self.valid_idx]

        # Scaling of X
        ss = StandardScaler()
        self.X_train_scaled = ss.fit_transform(self.X_train)
        self.X_valid_scaled = ss.transform(self.X_valid)
        self.X_test_scaled = ss.transform(self.X_test)

        # Clean Dataloaders
        self.train_dl = toy_utils.get_data_loader(self.X_train_scaled, self.y_train)
        self.valid_dl = toy_utils.get_data_loader(self.X_valid_scaled, self.y_valid)
        self.test_dl = toy_utils.get_data_loader(self.X_test_scaled, self.y_test)

        print('Training clean model...', end='', flush=True)
        self.model_clean = ToyInstanceNet(input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                                          output_dim=self.output_dim)
        if self.output_dim > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(self.model_clean.parameters(), lr=0.01)
        train(model=self.model_clean, train_loader=self.train_dl, val_loader=self.valid_dl, test_loader=self.test_dl,
              optimizer=optimizer, criterion=criterion, device=device, epochs=30, early_stopping=False, verbose=False)
        print('done!')

    def setup_noisy(self):
        # Rotation
        X_s = StandardScaler().fit_transform(self.X)
        self.X_noisy = toy_utils.rotate(X_s, theta=self.theta)

        # Training, Validation and Test
        self.X_train_noisy, self.X_test_noisy = self.X_noisy[self.train_idx_], self.X_noisy[self.test_idx]
        self.X_train_noisy, self.X_valid_noisy = self.X_train_noisy[self.train_idx], self.X_train_noisy[
            self.valid_idx]

        # Scaling of X noisy
        ss = StandardScaler()
        self.X_train_noisy_sc = ss.fit_transform(self.X_train_noisy)
        self.X_test_noisy_sc = ss.transform(self.X_test_noisy)
        self.X_valid_noisy_sc = ss.transform(self.X_valid_noisy)

        # Noisy Dataloaders
        self.train_noisy_dl = toy_utils.get_data_loader(self.X_train_noisy_sc, self.y_train)
        self.test_noisy_dl = toy_utils.get_data_loader(self.X_test_noisy_sc, self.y_test)
        self.valid_noisy_dl = toy_utils.get_data_loader(self.X_valid_noisy_sc, self.y_valid)

    def run(self, epochs):
        date_str = datetime.datetime.now().strftime('%H-%M-%S')

        ### TRAINING NOISY FROM SCRATCH ###
        print('Starting training from scratch using all samples...', end='', flush=True)
        self.model_scratch = ToyInstanceNet(input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                                            output_dim=self.output_dim)
        if self.output_dim > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(self.model_scratch.parameters(), lr=0.01)
        train_losses_scratch, train_accuracies_scratch, val_accuracies_scratch, \
            val_losses_scratch, epoch_scratch, _ = train(
                model=self.model_scratch,
                train_loader=self.train_noisy_dl,
                val_loader=self.test_noisy_dl,
                test_loader=self.valid_noisy_dl,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                epochs=epochs, early_stopping=False,
                verbose=False)
        print('done!')

        ### FINETUNING WITH ALL NOISY ###
        print('Starting finetuning using all samples...', end='', flush=True)
        self.model_all = copy.deepcopy(self.model_clean)
        if self.output_dim > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(self.model_all.parameters(), lr=0.01)
        train_losses_all, train_accuracies_all, val_accuracies_all, val_losses_all, epoch_all, _ = train(
            model=self.model_all,
            train_loader=self.train_noisy_dl,
            val_loader=self.valid_noisy_dl,
            test_loader=self.test_noisy_dl,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=epochs,
            early_stopping=False,
            verbose=False)
        print('done!')

        ### FINETUNING WITH SELECTED INDICES ###
        print('Starting training fine tuning using only selected samples...', end='', flush=True)
        self.model_selected_indices = copy.deepcopy(self.model_clean)
        if self.output_dim > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCELoss()
        optimizer = torch.optim.SGD(self.model_selected_indices.parameters(), lr=0.01)
        selected_indices = instance_selection(model=self.model_selected_indices, X_train=self.X_train_noisy_sc,
                                              y_train=self.y_train, sparse=False,
                                              X_valid=self.X_valid_noisy_sc, y_valid=self.y_valid,
                                              criterion=criterion, treshold=False)
        # selected_indices = instance_selection_no_hessian(model=self.model_selected_indices, X_train=self.X_train_noisy_sc,
        #                                                  y_train=self.y_train,
        #                                                  X_valid=self.X_valid_noisy_sc,
        #                                                  y_valid=self.y_valid, criterion=criterion)

        X_train_noisy_sel, y_train_sel = self.X_train_noisy_sc[selected_indices], self.y_train[selected_indices]
        train_noisy_dl_sel = toy_utils.get_data_loader(X_train_noisy_sel, y_train_sel)

        print('Number of selected samples = {}/{}'.format(len(selected_indices), len(self.X_train_noisy_sc)))
        train_losses_sel, train_accuracies_sel, val_accuracies_sel, val_losses_sel, epoch_sel, _ = train(
            model=self.model_selected_indices,
            train_loader=train_noisy_dl_sel,
            val_loader=self.valid_noisy_dl,
            test_loader=self.test_noisy_dl,
            optimizer=optimizer,
            criterion=criterion,
            device=device, epochs=epochs,
            early_stopping=False,
            verbose=False)
        print('done!')

        ### FINETUNING WITH RANDOM SAMPLES ###
        print('Starting finetuning using random samples...', end='', flush=True)
        rnd_indices = np.random.choice(len(self.X_train_noisy_sc), len(selected_indices), replace=False)
        X_train_noisy_rnd = self.X_train_noisy_sc[rnd_indices]
        y_train_rnd = self.y_train[rnd_indices]

        self.model_rnd = copy.deepcopy(self.model_clean)
        if self.output_dim > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCELoss()
        train_noisy_dl_rnd = toy_utils.get_data_loader(X_train_noisy_rnd, y_train_rnd)

        optimizer = torch.optim.SGD(self.model_rnd.parameters(), lr=0.01)
        train_losses_rnd, train_accuracies_rnd, val_accuracies_rnd, val_losses_rnd, epoch_rnd, _ = train(
            model=self.model_rnd,
            train_loader=train_noisy_dl_rnd,
            val_loader=self.valid_noisy_dl,
            test_loader=self.test_noisy_dl,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=epochs,
            early_stopping=False,
            verbose=False)
        print('done!')

        ### FINETUNING WITH NOT SELECTED SAMPLES ###
        print('Starting finetuning using not selected samples...', end='', flush=True)
        not_selected_indices = np.delete(range(len(self.X_train_noisy_sc)), selected_indices)
        X_train_noisy_not_sel = self.X_train_noisy_sc[not_selected_indices]
        y_train_not_sel = self.y_train[not_selected_indices]

        self.model_not_sel = copy.deepcopy(self.model_clean)
        if self.output_dim > 1:
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.BCELoss()
        train_noisy_dl_not_sel = toy_utils.get_data_loader(X_train_noisy_not_sel, y_train_not_sel)

        optimizer = torch.optim.SGD(self.model_not_sel.parameters(), lr=0.01)
        train_losses_not_sel, train_accuracies_not_sel, val_accuracies_not_sel, val_losses_not_sel, epoch_not_sel, _ = train(
            model=self.model_not_sel,
            train_loader=train_noisy_dl_not_sel,
            val_loader=self.valid_noisy_dl,
            test_loader=self.test_noisy_dl,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            epochs=epochs,
            early_stopping=False,
            verbose=False)
        print('done!')

        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot(range(epochs), train_accuracies_scratch, label='Scratch')
        plt.plot(range(epochs), train_accuracies_all, label='Finetuning with all samples')
        plt.plot(range(epochs), train_accuracies_sel, label='Finetuning with selected samples')
        plt.plot(range(epochs), train_accuracies_not_sel, label='Finetuning with not selected samples')
        plt.plot(range(epochs), train_accuracies_rnd, label='Finetuning with random samples')
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Train accuracy - Instance Selection\n')
        plt.legend(loc='best')

        plt.subplot(1, 3, 2)
        plt.plot(range(epochs), val_accuracies_scratch, label='Scratch')
        plt.plot(range(epochs), val_accuracies_all, label='Finetuning with all samples')
        plt.plot(range(epochs), val_accuracies_sel, label='Finetuning with selected samples')
        plt.plot(range(epochs), val_accuracies_not_sel, label='Finetuning with not selected samples')
        plt.plot(range(epochs), val_accuracies_rnd, label='Finetuning with random samples')
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Validation accuracy - Instance Selection\n')
        plt.legend(loc='best')
        plt.savefig(ROOT_DIR + '/toy-model/images/instance_based_selection_train_theta={}_{}.png'.format(self.theta, date_str))

        plt.subplot(1, 3, 3)
        plt.plot(range(epochs), val_losses_scratch, label='Scratch')
        plt.plot(range(epochs), val_losses_all, label='Finetuning with all samples')
        plt.plot(range(epochs), val_losses_sel, label='Finetuning with selected samples')
        plt.plot(range(epochs), val_losses_not_sel, label='Finetuning with not selected samples')
        plt.plot(range(epochs), val_losses_rnd, label='Finetuning with random samples')
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Validation loss - Instance Selection\n')
        plt.legend(loc='best')
        plt.savefig(ROOT_DIR + '/toy-model/images/instance_based_selection_train_theta={}_{}.png'.format(self.theta, date_str))
        plt.show()

        # plt.figure(figsize=(15, 5))
        # plt.subplot(1, 3, 1)
        # plt.scatter(self.X_train_scaled[:, 0], self.X_train_scaled[:, 1], marker='o', c=self.y_train,
        #             s=40, alpha=0.5, edgecolor='k')
        # plt.title('Total clean samples')
        #
        # plt.subplot(1, 3, 2)
        # plt.scatter(self.X_train_noisy_sc[:, 0], self.X_train_noisy_sc[:, 1], marker='o', c=self.y_train,
        #             s=40, alpha=0.5, edgecolor='k')
        # plt.title('Total noisy samples')
        #
        # plt.subplot(1, 3, 3)
        # plt.scatter(X_train_noisy_sel[:, 0], X_train_noisy_sel[:, 1], marker='o', c=y_train_sel,
        #             s=40, alpha=0.5, edgecolor='k')
        # plt.title('Selected noisy samples')
        # plt.savefig(ROOT_DIR + '/toy-model/images/instance_based_selection_points_theta={}_{}.png'.format(self.theta, date_str))
        # plt.show()


if __name__ == '__main__':
    for i in range(1):
        toy = ToyInstanceSelection(n_samples=5000, input_dim=20, hidden_dim=10, output_dim=1, theta=60.0)
        toy.run(50)

    exit()


    for i in range(1):
        toy = ToyInstanceSelection(n_samples=1000, input_dim=2, hidden_dim=3, output_dim=2, theta=70.0)
        toy.run(50)


    for i in range(1):
        toy = ToyInstanceSelection(n_samples=1000, input_dim=2, hidden_dim=3, output_dim=2, theta=50.0)
        toy.run(50)

    for i in range(1):
        toy = ToyInstanceSelection(n_samples=1000, input_dim=2, hidden_dim=3, output_dim=4, theta=50.0)
        toy.run(50)
