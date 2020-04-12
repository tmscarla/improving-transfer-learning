from toy_utils import *
from toy_models import *
from training import *
from toy_plots import *
from toy_models import FFSimpleNet
from toy_experiment import ToyExperiment
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import copy
from sampler import *
import numpy as np
import warnings
import torch

warnings.filterwarnings("ignore")


def get_leave_one_out_losses(toy, dset='valid'):
    """
    Givin a ToyExperiment instance, apply leave one out method to each sample of X_train_noisy.
    :param toy: an instance of ToyExperiment
    :return: losses: a list of losses with lenght equal to X_train_noisy
    """
    # Set seeds
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(0)
    torch.manual_seed(0)

    losses = []

    for i in tqdm(range(len(toy.X_train_noisy)), desc='Toy leave one out'):
        # Create new model
        model = FFSimpleNet(input_dim=toy.n_features, output_dim=1, activation='sigmoid')
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        # Create new training set without X[i]
        X_train_noisy = toy.X_train_noisy[np.arange(len(toy.X_train_noisy)) != i]
        y_train = toy.y_train[np.arange(len(toy.y_train)) != i]
        ss = StandardScaler()
        X_train_noisy_sc = ss.fit_transform(X_train_noisy)
        train_noisy_dl = get_data_loader(X_train_noisy_sc, y_train, shuffle=False)

        # Training
        train_losses, train_accuracies, val_accuracies, val_losses, epoch, model = train(model, train_noisy_dl,
                                                                                         toy.valid_dl, toy.test_dl,
                                                                                         optimizer=optimizer,
                                                                                         criterion=criterion,
                                                                                         device=device,
                                                                                         early_stopping=False,
                                                                                         epochs=toy.N_EPOCHS,
                                                                                         verbose=False)
        if dset == 'valid':
            losses.append(val_losses[-1])
        elif dset == 'train':
            losses.append(train_losses[-1])
    return losses


if __name__ == '__main__':
    toy = ToyExperiment(n_features=2, hidden_dim=2, n_classes=2, n_samples=100, theta=45, N_EPOCHS=10, convert_y=False)
    losses = get_leave_one_out_losses(toy)


