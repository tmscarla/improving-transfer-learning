import numpy as np
from training import *
from instance_based.models import InstanceMNISTNet
from downloads import load_MNIST, load_EMNIST
from utils import *
from constants import device


def training_model(model, dataset, split='letters', batch_size=128, epochs=5):
    if dataset == 'EMNIST':
        (X_train, y_train), (X_test, y_test) = load_EMNIST(split='letters')
    elif dataset == 'MNIST':
        (X_train, y_train), (X_test, y_test) = load_MNIST()
    else:
        raise Exception('Dataset not supported')

    print('Training on {}...'.format(dataset))

    # Split and scale
    (X_train, y_train), (X_valid, y_valid) = dataset_split(X_train, y_train, return_data='samples')

    X_train_sc, X_mean, X_std = image_preprocessing(X_train, scale_only=False)
    X_valid_sc, _, _ = image_preprocessing(X_valid, seq_mean=X_mean, seq_std=X_std, scale_only=False)
    X_test_sc, _, _ = image_preprocessing(X_test, seq_mean=X_mean, seq_std=X_std, scale_only=False)

    # Dataloaders
    train_dl = get_data_loader(X_train_sc, y_train, batch_size=batch_size)
    valid_dl = get_data_loader(X_valid_sc, y_valid, batch_size=batch_size)
    test_dl = get_data_loader(X_test_sc, y_test, batch_size=batch_size)

    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    train(model, train_dl, valid_dl, test_dl, optimizer=optimizer, device=device, criterion=criterion,
          early_stopping=False, epochs=epochs)


if __name__ == '__main__':
    ### EMNIST Training ###
    print('Training on EMNIST...')
    batch_size = 128
    model = InstanceMNISTNet(output_dim=26)
    training_MNIST(model=model, dataset='EMNIST', epochs=20)

    ### Optimized training ###

    ### MNIST finetuning ###
    model.fc2 = nn.Linear(32, 10)
    training_MNIST(model=model, dataset='MNIST', epochs=20)



