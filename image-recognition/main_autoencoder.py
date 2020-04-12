from autoencoder import *
from downloads import *
from constants import *
from experiments import Experiment
from utils import *
from datasets import *
from torch.utils.tensorboard import SummaryWriter
import os
import datetime




def check_autoencoder(path):
    """
    Load pre-trained autoencoder state from path and then display input and output images in one figure.
    :param path: .pt file of the state of the autoencoder
    """
    autoencoder = CIFAR10Autoencoder()
    autoencoder.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))

    (X_train, y_train), (X_test, y_test) = load_CIFAR10()
    X_train = image_preprocessing(X_train, scale_only=False)
    X_test = image_preprocessing(X_test, scale_only=False)

    train_loader = get_data_loader(X_train, y_train, shuffle=True)
    test_loader = get_data_loader(X_test, y_test, shuffle=False)

    for i, data in enumerate(test_loader):
        inputs, labels = data
        encoded, outputs = autoencoder(inputs)
        if i > 0: break

        for j, output in enumerate(outputs[:5]):
            plt.figure()

            # Display input
            plt.subplot(1, 2, 1)
            plt.title("Input | Label = {}".format(CIFAR_10_LABELS[labels[j].item()]))
            input_img = inputs[j].detach().numpy().transpose((1, 2, 0))
            plt.imshow(input_img)

            # Display output
            plt.subplot(1, 2, 2)
            plt.title("Output")
            output_img = output.detach().numpy().transpose((1, 2, 0))
            plt.imshow(output_img)
            plt.show()


if __name__ == '__main__':
    dataset = 'CIFAR_10'
    autoencoder = CIFAR10Autoencoder()

    from experiments import dataset_split
    # Load the data
    (X_train, y_train), (X_test, y_test) = load_CIFAR10()
    (X_train, y_train), (X_valid, y_valid) = dataset_split(X_train, y_train, return_data='samples')

    # Pre-processing the data
    X_train, _, _ = image_preprocessing(X_train, scale_only=False)
    X_valid, _, _ = image_preprocessing(X_valid, scale_only=False)
    X_test, _, _ = image_preprocessing(X_test, scale_only=False)

    # Data Loaders
    train_loader = get_data_loader(X_train, y_train, shuffle=True)
    val_loader = get_data_loader(X_valid, y_valid, shuffle=False)
    test_loader = get_data_loader(X_test, y_test, shuffle=False)

    # Writer and model path
    writer = SummaryWriter('runs/' + dataset + '_autoencoder')
    model_path = os.path.join(ROOT_DIR, MODELS_DIR, 'autoencoders')
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    model_path = os.path.join(model_path, 'CIFAR_10_autoencoder.pt')

    # Criterion and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)

    # Train
    train_autoencoder(autoencoder=autoencoder, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer,
                      criterion=criterion, device=device, writer=writer, start_epoch=None, scheduler=None, epochs=100,
                      early_stopping=True, model_path=model_path, flatten=False)
