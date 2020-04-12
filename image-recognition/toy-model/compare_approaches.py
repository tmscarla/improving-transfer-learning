import numpy as np
import warnings
import torch
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from toy_leave_one_out import get_leave_one_out_losses
from toy_property import compute_samples_property
from toy_experiment import ToyExperiment
from instance_based.instance_selection import instance_selection, instance_selection_no_hessian, \
    instance_selection_train_derivatives
from toy_utils import get_data_loader
from training import train, validate
from tqdm import tqdm
from datetime import datetime
from constants import ROOT_DIR
from toy_constants import device
import copy

N_EPOCHS = 30


def compare_approaches(save_fig=True):
    # Setup
    date = datetime.now().strftime("%m-%d_%H-%M-%S")
    toy = ToyExperiment(n_features=2, hidden_dim=5, n_classes=2, n_samples=500, theta=90, N_EPOCHS=20, convert_y=False)
    ss = StandardScaler()
    X_train_sc = ss.fit_transform(toy.X_train)
    X_train_noisy_sc = ss.fit_transform(toy.X_train_noisy)
    X_valid_noisy_sc = ss.fit_transform(toy.X_valid_noisy)

    # Compute loss and accuracy of clean model on noisy dataset
    train_dl = get_data_loader(X_train_noisy_sc, toy.y_train)
    valid_dl = get_data_loader(X_valid_noisy_sc, toy.y_valid)
    train_loss_clean, train_acc_clean = validate(toy.model_clean, train_dl, toy.criterion, device, flatten=True)
    valid_loss_clean, valid_acc_clean = validate(toy.model_clean, valid_dl, toy.criterion, device, flatten=True)

    # Percentage to keep
    perc = 0.5
    n_idx = int(len(X_train_sc) * perc)

    # Leave one out
    loo_losses = get_leave_one_out_losses(toy, dset='valid')
    loo_idx = np.argsort(loo_losses)
    loo_idx = loo_idx[::-1][:n_idx]
    # leave_one_out_idx = np.random.choice(len(X_train_noisy_sc), n_idx, replace=False)

    # Entropy
    entropy_idx = compute_samples_property(toy.model_clean, toy.X_train_noisy, toy.y_train, prop='entropy',
                                           flatten=True,
                                           indices=True, unique_labels=[0, 1])
    entropy_idx = entropy_idx[::-1][:n_idx]

    # Hessian
    hessian_losses = instance_selection(model=toy.model_clean, X_train=X_train_noisy_sc,
                                        y_train=toy.y_train, sparse=False,
                                        X_valid=X_valid_noisy_sc, y_valid=toy.y_valid,
                                        criterion=toy.criterion, treshold=False, return_influences=True)
    hessian_idx = np.argsort(hessian_losses)[:n_idx]

    # Jacobian
    jacobian_losses = instance_selection_no_hessian(model=toy.model_clean, X_train=X_train_noisy_sc,
                                                    y_train=toy.y_train,
                                                    X_valid=X_valid_noisy_sc, y_valid=toy.y_valid,
                                                    criterion=toy.criterion,
                                                    flatten=True, return_influences=True)
    jacobian_idx = np.argsort(jacobian_losses)[:n_idx]

    # Training derivatives
    train_derivatives = instance_selection_train_derivatives(model=toy.model_clean, X=X_train_noisy_sc,
                                                             y=toy.y_train,
                                                             criterion=toy.criterion,
                                                             flatten=True)

    train_derivatives_idx = np.argsort(train_derivatives)[:n_idx]

    # n_idx = min(len(hessian_idx), len(jacobian_idx))

    # Random
    random_idx = np.random.choice(len(X_train_noisy_sc), n_idx, replace=False)

    # Generate scatterplot
    methods = [(random_idx, 'random'), (loo_idx, 'leave one out'), (entropy_idx, 'entropy'),
               (hessian_idx, 'hessian'), (jacobian_idx, 'jacobian'), [train_derivatives_idx, 'train derivatives norms']]
    create_scatterplot(X_train_sc, X_train_noisy_sc, toy.y_train, X_valid_noisy_sc,
                       toy.y_valid, methods, n_idx, date=date)

    # Print common selected indices between loo and jacobian
    common = len(list(set(loo_idx) & set(jacobian_idx)))
    print('Common indices LOO and Jacobian: {}/{} | {:.2f}%'.format(common, n_idx, common / n_idx * 100))

    # Train
    plt.figure(figsize=(15, 5))

    for idx, name in tqdm(methods, desc='Training'):
        model_ = copy.deepcopy(toy.model_clean)
        X_train_ = X_train_noisy_sc[idx][:n_idx]
        y_train_ = toy.y_train[idx][:n_idx]
        optimizer = torch.optim.SGD(model_.parameters(), lr=0.01)

        train_dl_ = get_data_loader(X_train_, y_train_)

        train_losses_, train_accs_, val_accs_, val_losses_, epoch_, _ = train(model=model_, train_loader=train_dl_,
                                                                              val_loader=toy.valid_noisy_dl,
                                                                              test_loader=toy.test_noisy_dl,
                                                                              optimizer=optimizer,
                                                                              criterion=toy.criterion,
                                                                              device=device, epochs=N_EPOCHS,
                                                                              early_stopping=False,
                                                                              flatten=True,
                                                                              verbose=False)
        # Make accuracies and losses starting from the same intial point
        train_losses_.insert(0, train_loss_clean)
        train_accs_.insert(0, train_acc_clean)
        val_losses_.insert(0, valid_loss_clean)
        val_accs_.insert(0, valid_acc_clean)

        # Plot
        plt.subplot(1, 2, 1)
        plt.plot(range(N_EPOCHS + 1), train_accs_, label=name)
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Train\n')
        plt.legend(loc='best')
        plt.subplot(1, 2, 2)
        plt.plot(range(N_EPOCHS + 1), val_accs_, label=name)
        plt.xlabel('Epochs', fontsize=16)
        plt.ylabel('Accuracy', fontsize=16)
        plt.title('Validation\n')
        plt.legend(loc='best')

    if save_fig:
        plt.savefig(ROOT_DIR + '/toy-model/images/compare_{}_accuracy.png'.format(date))
    plt.show()


def create_scatterplot(X_train_sc, X_train_noisy_sc, y_train, X_valid_noisy_sc,
                       y_valid, methods, n_idx, save_fig=True, date=""):
    # Train and validation
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(X_train_sc[:, 0], X_train_sc[:, 1], marker='o', c=y_train,
                s=40, alpha=0.7, edgecolor='k')

    plt.xlim(right=3.0)  # xmax is your value
    plt.xlim(left=-3.0)  # xmin is your value
    plt.ylim(top=3.0)  # ymax is your value
    plt.ylim(bottom=-3.0)  # ymin is your value
    plt.title('Train clean')

    plt.subplot(1, 3, 2)
    plt.scatter(X_train_noisy_sc[:, 0], X_train_noisy_sc[:, 1], marker='o', c=y_train,
                s=40, alpha=0.7, edgecolor='k')

    plt.xlim(right=3.0)  # xmax is your value
    plt.xlim(left=-3.0)  # xmin is your value
    plt.ylim(top=3.0)  # ymax is your value
    plt.ylim(bottom=-3.0)  # ymin is your value
    plt.title('Train noisy')

    plt.subplot(1, 3, 3)
    plt.scatter(X_valid_noisy_sc[:, 0], X_valid_noisy_sc[:, 1], marker='o', c=y_valid,
                s=40, alpha=0.7, edgecolor='k')

    plt.xlim(right=3.0)  # xmax is your value
    plt.xlim(left=-3.0)  # xmin is your value
    plt.ylim(top=3.0)  # ymax is your value
    plt.ylim(bottom=-3.0)  # ymin is your value
    plt.title('Validation noisy')

    if save_fig:
        plt.savefig(ROOT_DIR + '/toy-model/images/compare_{}_rotation.png'.format(date))
    plt.show()

    # Approaches
    plt.figure(figsize=(25, 5))
    for i, (idx, name) in enumerate(methods):
        plt.subplot(1, 6, i + 1)
        plt.scatter(X_train_noisy_sc[idx][:n_idx][:, 0], X_train_noisy_sc[idx][:n_idx][:, 1], marker='o',
                    c=y_train[idx][:n_idx], s=40, alpha=0.7, edgecolor='k')

        plt.xlim(right=3.0)  # xmax is your value
        plt.xlim(left=-3.0)  # xmin is your value
        plt.ylim(top=3.0)  # ymax is your value
        plt.ylim(bottom=-3.0)  # ymin is your value
        plt.title(name)

    if save_fig:
        plt.savefig(ROOT_DIR + '/toy-model/images/compare_{}_scatterplot.png'.format(date))
    plt.show()


if __name__ == '__main__':
    compare_approaches()
