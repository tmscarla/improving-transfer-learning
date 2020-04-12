import torchvision
from torch.utils.tensorboard import SummaryWriter
import torch

from datasets import get_data_loader
from downloads import load_dataset
from training import validate, train, train_baseline
from utils import load_dataloaders_from_dataset, dataset_split, image_preprocessing
from shortcuts import setup_finetuning
from constants import *
import copy
import numpy as np
from tqdm import tqdm
from property import compute_samples_property


def importance(model, X_train, y_train, optimizer, criterion, epochs, device,
               val_dl, prop, writer=None, early_stopping=True, batch_size=64, recompute_epoch=0):

    # Compute initial importance values
    unique_labels = len(list(set(y_train)))
    importances = compute_samples_property(model, X_train, y_train, prop=prop,
                                           unique_labels=unique_labels,
                                           indices=False, verbose=True)

    # Initialization
    model.train()
    curr_epoch, size = 0, len(importances)
    rnd_indices = np.random.randint(0, len(X_train), batch_size)
    running_loss, running_corrects, total = 0, 0, 0
    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []
    if recompute_epoch == 0: recompute_epoch = epochs

    # Simulate an epoch using len(X_train) = batch_size * batches
    while curr_epoch < epochs:
        if curr_epoch % recompute_epoch == 0 and curr_epoch != 0:
            importances = compute_samples_property(model, X_train, y_train, prop=prop,
                                                   unique_labels=unique_labels,
                                                   indices=False, verbose=True)

        seen_samples, running_loss = 0, 0
        while seen_samples < len(X_train):
            if curr_epoch == 0:
                curr_indices = rnd_indices
            else:
                curr_indices = metropolis_hastings(importances, curr_indices)

            inputs, labels = X_train[curr_indices], y_train[curr_indices]
            inputs = np.moveaxis(inputs, source=-1, destination=1).astype(np.float32)
            inputs, labels = torch.Tensor(inputs), torch.LongTensor(labels)
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            labels = labels.long()
            running_corrects += (predicted == labels).sum().item()
            running_loss += loss
            total += labels.size(0)

            seen_samples += len(curr_indices)

        # Train loss/accuracy
        train_loss = running_loss
        train_acc = running_corrects / total
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # Validation loss/accuracy
        val_loss, val_acc = validate(model, val_dl, criterion, device, flatten=False)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        curr_epoch += 1
        print('[ EPOCH {}: train_loss={} | train_acc={} | val_loss={} | val_acc={} ]'.format(curr_epoch, train_loss,
                                                                                             train_acc, val_loss,
                                                                                             val_acc))


def metropolis_hastings(importances, curr_indices):
    selected_indices = np.random.randint(0, len(importances), len(curr_indices))

    for i in range(len(selected_indices)):
        t = np.random.rand()
        curr_importance = importances[curr_indices[i]]
        old_importance = importances[selected_indices[i]]

        if old_importance == 0:  # Avoid division by zero
            continue
        f = curr_importance / old_importance
        if t > f:
            curr_indices[i] = selected_indices[i]

    return curr_indices


def compute_losses(X, y, model, criterion):
    losses = []
    for i, x in enumerate(tqdm(X, desc='Compute losses')):
        x = torch.Tensor(x).view(1, *x.shape)
        label = y[i]
        label = torch.LongTensor([label])
        output = model(x)
        loss = criterion(output, label)
        losses.append(loss)

    return np.array(losses)


if __name__ == '__main__':
    # model = 'VGG'
    # dataset = 'CIFAR-10'
    # pretrained = True
    #
    # vggnet = torchvision.models.vgg11_bn(pretrained=pretrained, progress=True)
    # vggnet.classifier = torch.nn.Linear(512, 10)
    #
    # train_dl, val_dl, test_dl = load_dataloaders_from_dataset(dataset)
    # (X_train, y_train), (X_test, y_test) = load_dataset(dataset)
    # optimizer = torch.optim.Adam(vggnet.parameters())
    # criterion = torch.nn.CrossEntropyLoss()
    # writer = SummaryWriter('runs/model={}_pretrained={}_dset={}'.format(model, pretrained, dataset))
    #
    # importance(vggnet, X_train, y_train, train_dl, val_dl, test_dl, optimizer, criterion, 100, device)

    ##############################

    baseline, classes, dataset, noise = 'SimpleBaselineNet', list(range(10)), 'CIFAR-10', 'AWGN'
    # train_baseline(dataset, 'SimpleBaselineNet', noisy=False, classes=classes)
    model_clean, (X_train_noisy, X_test_noisy,
                  X_train, X_test, y_train, y_test) = setup_finetuning(baseline, classes, dataset, noise, mean=0.0,
                                                                       std=15.0)
    print('Loaded baseline!')

    # Validation splitting
    (X_train_noisy, y_train), (X_valid_noisy, y_valid) = dataset_split(X_train_noisy, y_train,
                                                                       return_data='samples')

    # Image pre-processing: scale pixel values
    X_train_noisy_sc, X_mean, X_std = image_preprocessing(X_train_noisy, scale_only=False)
    X_valid_noisy_sc, _, _ = image_preprocessing(X_valid_noisy, seq_mean=X_mean, seq_std=X_std, scale_only=False)
    X_test_noisy_sc, _, _ = image_preprocessing(X_test_noisy, seq_mean=X_mean, seq_std=X_std, scale_only=False)

    # Dataloaders
    train_noisy_dl = get_data_loader(X_train_noisy_sc, y_train, shuffle=True)
    valid_noisy_dl = get_data_loader(X_valid_noisy_sc, y_valid, shuffle=True)
    test_noisy_dl = get_data_loader(X_test_noisy_sc, y_test, shuffle=True)

    # Writer
    writer = SummaryWriter('runs/' + '{}_{}_fine_tuning'.format(baseline, dataset))

    # Fine-tuning
    print('Fine-tuning...')
    model_finetune = copy.deepcopy(model_clean)
    n_classes = len(np.unique(y_train))
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model_finetune.parameters())
    train_losses, train_accuracies, val_accuracies, val_losses, _, _ = train(model_finetune, train_noisy_dl,
                                                                             valid_noisy_dl, test_noisy_dl, optimizer,
                                                                             criterion, device, epochs=50,
                                                                             early_stopping=False)

    for prop in ['entropy', 'cross_entropy', 'first_vs_second']:
        for batch_size in [64, 128, 256]:
            for rec_epoch in [5, 10, 20]:
                # Importance sampling
                print('Importance sampling...')
                model_importance = copy.deepcopy(model_clean)
                optimizer = torch.optim.Adam(model_importance.parameters())
                criterion = torch.nn.CrossEntropyLoss()
                writer = SummaryWriter('runs/' + '{}_{}_importance_sampling_prop={}_bs={}_re={}'.format(baseline,
                                                                                                        dataset, prop,
                                                                                                        batch_size, rec_epoch))

                importance(model_importance, X_train_noisy_sc, y_train, optimizer, criterion,
                           epochs=50, device=device, writer=writer, val_dl=valid_noisy_dl,
                           prop=prop, recompute_epoch=rec_epoch, batch_size=batch_size)
