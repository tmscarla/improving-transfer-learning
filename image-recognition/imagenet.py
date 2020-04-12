import torch
import numpy as np
import torchvision
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.tensorboard import SummaryWriter

from datasets import image_preprocessing, get_data_loader
from downloads import load_dataset
from constants import *
from training import train
import torch
from utils import load_dataloaders_from_dataset

if __name__ == '__main__':
    model = 'VGG'
    dataset = 'CIFAR-100'
    pretrained = True

    vggnet = torchvision.models.vgg11_bn(pretrained=pretrained, progress=True)
    vggnet.classifier = torch.nn.Sequential(torch.nn.Linear(in_features=25088, out_features=1024, bias=True),
                                            torch.nn.ReLU(inplace=True),
                                            torch.nn.Dropout(p=0.5, inplace=False),
                                            torch.nn.Linear(1024, 100))

    train_dl, val_dl, test_dl = load_dataloaders_from_dataset(dataset)
    optimizer = torch.optim.SGD(vggnet.parameters(), lr=0.01,
                                momentum=0.9,
                                weight_decay=1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    writer = SummaryWriter('runs/model={}_pretrained={}_dset={}'.format(model, pretrained, dataset))

    train(vggnet, train_dl, val_dl, test_loader=test_dl, optimizer=optimizer, criterion=criterion,
          device=device, epochs=5000, early_stopping=True, writer=writer)


