import torch
import numpy as np
from training_NNLosc import train_NNLosc_bundle
from constants import *
from alternative_solvers import NLosc_solution
import matplotlib.pyplot as plt


class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)


class odeNet_NLosc_bundle(torch.nn.Module):
    def __init__(self, input=3, hidden=50):
        super(odeNet_NLosc_bundle, self).__init__()
        self.activation = mySin()
        self.fc1 = torch.nn.Linear(input, hidden)
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

    def parametric_solution(self, t, initial_conditions):
        t_0, x_0, px_0, lam = initial_conditions[0], initial_conditions[1], initial_conditions[2], initial_conditions[3]

        xs, ps = [x_0] * len(t), [px_0] * len(t)
        xs, ps = torch.FloatTensor(xs), torch.FloatTensor(ps)
        xs = xs.view((len(t), 1))
        ps = ps.view((len(t), 1))

        bundle = torch.cat([t, xs, ps], dim=1)
        N1, N2 = self.forward(bundle)

        dt = t - t_0
        f = (1 - torch.exp(-dt))

        x_hat = x_0 + f * N1
        px_hat = px_0 + f * N2
        return x_hat, px_hat


def bundle_vs_solver(model, initial_conditions, t_0, t_final, size):
    t = torch.linspace(t_0, t_final, size).reshape(-1, 1)
    x, p = model.parametric_solution(t, initial_conditions)
    x_hat, p_hat = x.detach().numpy(), p.detach().numpy()
    t = t.detach().numpy().flatten()
    x_true, p_true = NLosc_solution(t, initial_conditions[1], initial_conditions[2], initial_conditions[3])

    fig = plt.figure(figsize=(15, 5))
    st = fig.suptitle('Starting conditions: x(0) = 1.1 | p(0) = 1.1\n'
                      'Model trained with x(0) and p(0) in bundle [1.0, 1.2]')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.plot(t, x_true, linewidth=1, color='#cc0000', label='scipy')
    ax1.plot(t, x_hat, linewidth=1, color='#3366ff', label='network')
    ax1.set_xlabel('t', fontsize=12)
    ax1.set_ylabel('x', fontsize=12)
    ax1.legend()
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel('t', fontsize=12)
    ax2.set_ylabel('p', fontsize=12)
    ax2.plot(t, p_true, linewidth=1, color='#cc0000', label='scipy')
    ax2.plot(t, p_hat, linewidth=1, color='#3366ff', label='network')
    ax2.legend()

    fig.tight_layout()

    st.set_y(0.95)
    fig.subplots_adjust(top=0.82)

    plt.savefig('img.png')
    fig.show()


if __name__ == '__main__':
    model = odeNet_NLosc_bundle()
    t_0, t_final, train_size = 0., 4 * np.pi, 200
    lam = 1
    epochs = int(5e4)
    x_l, x_r = 2.2, 2.5
    px_l, px_r = 2.0, 2.2

    optimizer = torch.optim.Adam(model.parameters(), lr=8e-4)

    # Train and save
    train_NNLosc_bundle(model, x_l, x_r, px_l, px_r, t_0, t_final, lam, epochs, optimizer, train_size)
    torch.save({'model_state_dict': model.state_dict()},
               ROOT_DIR + '/models/NLosc/bundle/x_l={}'
                          '-px_l={}-x_r={}-px_r={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_l, px_l, x_r, px_r,
                                                                                       t_0, t_final, lam))
    exit()
    # Load
    path = ROOT_DIR + '/models/NLosc/bundle/x_l={}-px_l={}-x_r={}-px_r={}-t_0={}-t_f={:.2f}_lam={}.pt' \
        .format(x_l, px_l, x_r, px_r, t_0, t_final, lam)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Select one condition inside the bundle
    # x_i, px_i = 1.1, 1.1
    # bundle_vs_solver(model, [t_0, x_i, px_i, lam], t_0, t_final, train_size)

    # Finetuning
    print('Finetuning...')
    fine_model = odeNet_NLosc_bundle()
    path = ROOT_DIR + '/models/NLosc/bundle/x_l={}-px_l={}-x_r={}-px_r={}-t_0={}-t_f={:.2f}_lam={}.pt' \
        .format(x_l, px_l, x_r, px_r, t_0, t_final, lam)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    fine_model.load_state_dict(checkpoint['model_state_dict'])
    t_0, t_final, train_size = 0., 4 * np.pi, 200
    lam = 1
    epochs = int(5e4)
    x_l, x_r = 2.2, 2.5
    px_l, px_r = 2.0, 2.2

    optimizer = torch.optim.Adam(fine_model.parameters(), lr=8e-4)
    train_NNLosc_bundle(fine_model, x_l, x_r, px_l, px_r, t_0, t_final, lam, epochs, optimizer,
                        train_size, additional_comment='_bundle_finetuning')
