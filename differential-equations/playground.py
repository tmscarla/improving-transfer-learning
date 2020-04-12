import torch
import numpy as np
from training_NNLosc import train_NNLosc
from generate_plots import plot_single_NLosc, plot_multiple_NLosc
import copy
from constants import *


class STR(torch.nn.Module):
    def __init__(self):
        def st_hook(grad):
            out = grad.clone()
            out[0][1] = 0
            out[1][0] = 0
            return out

        def r_hook(grad):
            out = grad.clone()
            out[0][1] = 0
            out[1][0] = 0
            out[1][1] = 0
            return out

        super(STR, self).__init__()
        self.st = torch.nn.Linear(in_features=2, out_features=2, bias=True)
        self.st.weight.data[0][1] = 0
        self.st.weight.data[1][0] = 0
        self.st.register_backward_hook(st_hook)

        self.r = torch.nn.Linear(in_features=2, out_features=2, bias=False)
        self.r.register_backward_hook(r_hook)

    def forward(self, x):
        x = self.st(x)
        x = self.r(x)

        return x


class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)


class odeNet_NLosc_STR(torch.nn.Module):
    def __init__(self, hidden=10, tau=0):
        super(odeNet_NLosc_STR, self).__init__()
        self.tau = tau
        self.activation = mySin()
        self.fc1 = torch.nn.Linear(1, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.out = torch.nn.Linear(hidden, 2)

        self.st = torch.nn.Linear(2, 2)
        self.st.weight.data[0][1] = 0
        self.st.weight.data[1][0] = 0
        self.st.weight.data[0][0] = 1
        self.st.weight.data[1][1] = 1
        self.st.bias.data[0] = 0
        self.st.bias.data[1] = 0
        self.st.weight.requires_grad = False
        self.st.bias.requires_grad = False

        self.r = torch.nn.Linear(2, 2)
        self.r.weight.data[0][1] = 0
        self.r.weight.data[1][0] = 0
        self.r.weight.data[0][0] = 1
        self.r.weight.data[1][1] = 1
        self.r.bias.data[0] = 0
        self.r.bias.data[1] = 0
        self.r.weight.requires_grad = False
        self.r.bias.requires_grad = False

        self.rotation_matrix = None

    def forward(self, z):
        z = self.fc1(z)
        z = self.activation(z)
        z = self.fc2(z)
        z = self.activation(z)
        z = self.out(z)
        z = self.st(z)
        z = self.r(z)

        # TODO Rotate
        if self.rotation_matrix is not None:
            from analytical import transform
            z = transform(z, self.rotation_matrix)

        # Split the output in coordinates and momentum
        x_N = (z[:, 0]).reshape(-1, 1)
        px_N = (z[:, 1]).reshape(-1, 1)

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


if __name__ == '__main__':
    str = odeNet_NLosc_STR()
    x_0, px_0, lam = 1.0, 1.0, 1
    t_0, t_final, train_size = 0., 4 * np.pi, 200
    initial_conditions = [t_0, x_0, px_0, lam]
    model = odeNet_NLosc_STR(hidden=50)
    betas = (0.999, 0.9999)
    lr = 8e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)

    # Train and save
    # _, losses, _, _ = train_NNLosc(initial_conditions, t_final, lam, hidden=50,
    #                                epochs=int(3e4), train_size=200, optimizer=optimizer, start_model=source_model)
    # torch.save({'model_state_dict': source_model.state_dict()},
    #            ROOT_DIR + '/models/NLosc/STR/x_0={}'
    #                       '-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_0, px_0, t_0, t_final, lam))
    # print('Loss pre:', losses[-1])

    # Load
    path = ROOT_DIR + '/models/NLosc/STR/x_0={}-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_0, px_0, t_0, t_final,
                                                                                            lam)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])

    pre_model = copy.deepcopy(model)

    # Train ST
    # source_model.fc1.weight.requires_grad = False
    # source_model.fc1.bias.requires_grad = False
    # source_model.fc2.weight.requires_grad = False
    # source_model.fc2.bias.requires_grad = False
    # source_model.out.weight.requires_grad = False
    # source_model.out.bias.requires_grad = False

    # source_model.st.weight.requires_grad = True
    # source_model.st.bias.requires_grad = True
    # source_model.r.weight.requires_grad = True
    # source_model.r.bias.requires_grad = True

    # source_model.r.weight.data[0][1] = np.sqrt(3) / 2
    # source_model.r.weight.data[1][0] = (-1) * np.sqrt(3) / 2
    # source_model.r.weight.data[0][0] = 0.5
    # source_model.r.weight.data[1][1] = 0.5

    x_e, px_e, lam = 2.0, 2.0, 1
    end_conditions = [t_0, x_e, px_e, lam]
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    from analytical import define_A
    model.rotation_matrix = define_A(x_0, px_0, x_e, px_e, rotation_only=True)

    _, losses, _, _ = train_NNLosc(end_conditions, t_final, lam, hidden=50, epochs=int(5e4),
                                   train_size=200, optimizer=optimizer, start_model=model, str=False,
                                   num_batches=1, additional_comment='_analytical')
    #plot_single_NLosc(source_model, end_conditions, t_0, t_final, train_size)
    print('Loss pre+st:', losses[-1])

    exit()

    # Scratch
    scratch = odeNet_NLosc_STR(hidden=50)
    betas = (0.999, 0.9999)
    lr = 8e-4
    optimizer = torch.optim.Adam(scratch.parameters(), lr=lr, betas=betas)

    # Train and save
    # _, losses, _, _ = train_NNLosc(end_conditions, t_final, lam, hidden=50, epochs=int(4e4),
    #                                train_size=200, optimizer=optimizer, start_model=scratch)
    # torch.save({'model_state_dict': scratch.state_dict()},
    #            ROOT_DIR + '/models/NLosc/STR/x_0={}'
    #                       '-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_e, px_e, t_0, t_final, lam))
    # Load
    path = ROOT_DIR + '/models/NLosc/STR/x_0={}-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_e, px_e, t_0, t_final,                                                                          lam)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    scratch.load_state_dict(checkpoint['model_state_dict'])

    plot_multiple_NLosc([pre_model, model, scratch], ['init', 'init+st', 'end'],
                        [initial_conditions, end_conditions, end_conditions], t_0,
                        t_final, 200)
    # print('Loss end:', losses[-1])
