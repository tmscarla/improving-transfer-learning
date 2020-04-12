import torch.nn as nn


class FFSimpleNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=3, output_dim=1, activation='tanh'):
        super(FFSimpleNet, self).__init__()
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, output_dim)
        if activation == 'tanh':
            self.r1 = nn.Tanh()
        elif activation == 'sigmoid':
            self.r1 = nn.Sigmoid()
        elif activation == 'softmax':
            self.r1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.r1(x)
        return x


class ToyInstanceNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=3, output_dim=3):
        super(ToyInstanceNet, self).__init__()
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, output_dim)

        if output_dim == 1:
            self.out = nn.Sigmoid()
        else:
            self.out = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.out(x)
        return x
