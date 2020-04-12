import torch
import numpy as np

# Define the sin() activation function
class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)


class SIRNetwork(torch.nn.Module):
    def __init__(self, activation=None, input=1, layers=2, hidden=10, output=3):
        super(SIRNetwork, self).__init__()
        if activation is None:
            self.activation = torch.nn.Sigmoid()
        else:
            self.activation = activation

        self.fca = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            self.activation
        )
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(input, hidden),
            *[self.fca for _ in range(layers)],
            torch.nn.Linear(hidden, output)
        )

    def forward(self, x):
        x = self.ffn(x)
        s_N = (x[:, 0]).reshape(-1, 1)
        i_N = (x[:, 1]).reshape(-1, 1)
        r_N = (x[:, 2]).reshape(-1, 1)

        return s_N, i_N, r_N

    def parametric_solution(self, t, initial_conditions, beta=None, gamma=None, mode=None, noise_mean=0, noise_std=0):
        # Parametric solutions
        if mode is None or mode == 'bundle_params':
            t_0 = initial_conditions[0]
            s_0, i_0, r_0 = initial_conditions[1]
        else:
            t_0 = 0
            s_0, i_0, r_0 = initial_conditions[0][:], initial_conditions[1][:], torch.zeros(
                initial_conditions[0][:].shape)

        # If beta and gamma are not None, we are trying to learn a bundle
        if mode is None:
            N1, N2, N3 = self.forward(t)
        elif mode == 'bundle_init':
            t_bundle = torch.cat([t, s_0], dim=1)
            N1, N2, N3 = self.forward(t_bundle)
        elif mode == 'bundle_params':
            t_bundle = torch.cat([t, beta, gamma], dim=1)
            N1, N2, N3 = self.forward(t_bundle)
        elif mode == 'bundle_total':
            t_bundle = torch.cat([t, s_0, beta, gamma], dim=1)
            N1, N2, N3 = self.forward(t_bundle)

        dt = t - t_0

        f = (1 - torch.exp(-dt))

        s_hat = (s_0 + f * (N1))
        i_hat = (i_0 + f * (N2))
        r_hat = (r_0 + f * (N3))

        return s_hat, i_hat, r_hat


# A two hidden layer Neural Network, 1 input & two output
class odeNet_NLosc_MM(torch.nn.Module):
    def __init__(self, hidden=10, tau=0):
        super(odeNet_NLosc_MM, self).__init__()
        self.tau = tau
        self.activation = mySin()
        self.fc1 = torch.nn.Linear(1, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.out = torch.nn.Linear(hidden, 2)


    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.out(x)

        # Split the output in coordinates and momentum
        x_N = (x[:, 0]).reshape(-1, 1)
        px_N = (x[:, 1]).reshape(-1, 1)
        return x_N, px_N

    def parametric_solution(self, t, X_0):
        # Parametric solutions
        t_0, x_0, px_0, lam = X_0[0], X_0[1], X_0[2], X_0[3]
        N1, N2 = self.forward(t + self.tau)
        dt = t - t_0

        f = (1 - torch.exp(-dt))

        x_hat = x_0 + f * N1
        px_hat = px_0 + f * N2
        return x_hat, px_hat


class odeNet_NLosc_modular(torch.nn.Module):
    def __init__(self, hidden=10, layers=2):
        super(odeNet_NLosc_modular, self).__init__()
        self.activation = mySin()
        self.fca = torch.nn.Sequential(
            torch.nn.Linear(hidden, hidden),
            self.activation
        )
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(1, hidden),
            *[self.fca for _ in range(layers)],
            torch.nn.Linear(hidden, 2)
        )

    def forward(self, x):
        x = self.ffn(x)

        # Split the output in coordinates and momentum
        x_N = (x[:, 0]).reshape(-1, 1)
        px_N = (x[:, 1]).reshape(-1, 1)
        return x_N, px_N

    def parametric_solution(self, t, X_0):
        # Parametric solutions
        t_0, x_0, px_0, lam = X_0[0], X_0[1], X_0[2], X_0[3]
        N1, N2 = self.forward(t)
        dt = t - t_0

        f = (1 - torch.exp(-dt))

        x_hat = x_0 + f * N1
        px_hat = px_0 + f * N2
        return x_hat, px_hat
