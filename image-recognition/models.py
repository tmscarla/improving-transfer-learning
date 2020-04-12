import torch
from torch import nn
import numpy as np
from torch.functional import F
import time
import os
import sys

window_size = 8


class SimpleBaselineNet(nn.Module):
    def __init__(self, output_dim=10):
        super(SimpleBaselineNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_features=32)  # Equals the number of the previous output channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_features=16)  # Equals the number of the previous output channels
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(16 * window_size * window_size, 128)
        self.fc1_dp = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(128, output_dim)
        self.out = nn.Softmax(dim=1)

    def forward(self, x):

        x = self.pool1(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool2(F.relu(self.conv2_bn(self.conv2(x))))

        x = x.view(-1, 16 * window_size * window_size)
        x = F.relu(self.fc1(x))
        x = self.fc1_dp(x)
        x = self.fc2(x)
        x = self.out(x)

        return x


class SimpleBaselineBinaryNet(nn.Module):
    def __init__(self, activation='sigmoid', num_conv=32, num_ff=128):
        super(SimpleBaselineBinaryNet, self).__init__()
        self.num_conv = num_conv

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=num_conv, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_features=num_conv)  # equals the number of the previous output channels
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(in_channels=num_conv, out_channels=num_conv//2, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_features=num_conv//2)  # equals the number of the previous output channels
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(num_conv//2 * window_size * window_size, num_ff)
        self.fc1_dp = nn.Dropout2d(p=0.5)
        self.fc2 = nn.Linear(num_ff, 1)
        if activation == 'sigmoid':
            self.out = nn.Sigmoid()
        elif activation == 'tanh':
            self.out = nn.Tanh()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool2(F.relu(self.conv2_bn(self.conv2(x))))

        x = x.view(-1, self.num_conv//2 * window_size * window_size)
        x = F.relu(self.fc1(x))
        x = self.fc1_dp(x)
        x = self.fc2(x)
        x = self.out(x)

        return x

    def forward_features(self, x):
        x = self.pool1(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool2(F.relu(self.conv2_bn(self.conv2(x))))

        x = x.view(-1, 16 * window_size * window_size)
        x = F.relu(self.fc1(x))

        return x


class ACNBaselineNet(nn.Module):
    def __init__(self):
        super(ACNBaselineNet, self).__init__()

        self.img_dp = nn.Dropout2d(p=0.2)

        # One padding layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.conv1_bn = nn.BatchNorm2d(num_features=96)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=1)
        self.conv2_bn = nn.BatchNorm2d(num_features=96)
        self.conv3 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1)
        self.conv3_bn = nn.BatchNorm2d(num_features=96)

        # Dropout is applied after each layer replacing the max pooling operation
        self.mp1_dp = nn.Dropout2d(p=0.5)
        self.conv4 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.conv4_bn = nn.BatchNorm2d(num_features=192)
        self.conv5= nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.conv5_bn = nn.BatchNorm2d(num_features=192)
        self.conv6 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=2, padding=1)
        self.conv6_bn = nn.BatchNorm2d(num_features=192)

        # Dropout is applied after each layer replacing the max pooling operation
        self.mp2_dp = nn.Dropout2d(p=0.5)

        # Zero padding layers
        self.conv7 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=0)
        self.conv7_bn = nn.BatchNorm2d(num_features=192)
        self.conv8 = nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0)
        self.conv8_bn = nn.BatchNorm2d(num_features=192)
        self.conv9 = nn.Conv2d(in_channels=192, out_channels=100, kernel_size=1, stride=1, padding=0)
        self.conv9_bn = nn.BatchNorm2d(num_features=100)

        self.out = nn.Softmax(dim=1)

    def forward(self, x):
        # Dropout on image input
        x = self.img_dp(x)

        # Relu after batch_norm
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))

        # Dropout on max pool layer
        x = self.mp1_dp(x)

        # Relu after batch_norm
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))

        # Dropout on max pool layer
        x = self.mp2_dp(x)

        # Relu after batch_norm
        x = F.relu(self.conv7_bn(self.conv7(x)))
        x = F.relu(self.conv8_bn(self.conv8(x)))
        x = F.relu(self.conv9_bn(self.conv9(x)))

        # X here has shape (batch_size, num_features, w, h) which is (64, 100, 1, 1)

        # Global average pooling for each feature map (mean over the flattened feature map dimension)
        x = torch.mean(x.view(x.size(0), x.size(1), -1), dim=2)

        x = self.out(x)
        return x


class FFSimpleNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=16, output_dim=10):
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


class FFBinaryNet(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=3, output_dim=1):
        super(FFBinaryNet, self).__init__()
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.r1 = nn.ReLU()
        self.f2 = nn.Linear(hidden_dim, output_dim)
        self.out = nn.Sigmoid()

    def forward(self, x):
        x = self.f1(x)
        x = self.r1(x)
        x = self.f2(x)
        x = self.out(x)
        return x
