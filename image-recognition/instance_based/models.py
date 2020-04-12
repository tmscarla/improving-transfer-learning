import torch
from torch import nn
from torch.functional import F
import numpy as np
import time
import os
import sys


class InstanceMNISTNet(nn.Module):
    def __init__(self, output_dim):
        super(InstanceMNISTNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv1.weight, gain=1.0)
        self.conv1_bn = nn.BatchNorm2d(num_features=16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        nn.init.xavier_uniform_(self.conv2.weight, gain=1.0)
        self.conv2_bn = nn.BatchNorm2d(num_features=32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # self.conv3 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        # nn.init.xavier_uniform_(self.conv3.weight, gain=1.0)
        # self.conv3_bn = nn.BatchNorm2d(num_features=32)
        #
        # self.conv4 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        # nn.init.xavier_uniform_(self.conv4.weight, gain=1.0)
        # self.conv4_bn = nn.BatchNorm2d(num_features=64)
        # self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(6272, 32)
        self.fc2 = nn.Linear(32, output_dim)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = self.pool2(x)

        # x = F.relu(self.conv3_bn(self.conv3(x)))
        # x = F.relu(self.conv4_bn(self.conv4(x)))
        # x = self.pool4(x)

        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)

        return x


class FFSimpleNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=3, output_dim=10):
        super(FFSimpleNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.r1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.r1(x)
        x = self.fc2(x)
        x = self.out(x)
        return x
