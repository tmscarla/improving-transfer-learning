import torch
import numpy as np
from training_NNLosc import train_NNLosc, train_NNLosc_points, train_NNLosc_Mblock
from generate_plots import plot_single_NLosc, plot_multiple_NLosc
import copy
from constants import *
from models import odeNet_NLosc_MM
from analytical import define_M
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt


class mySin(torch.nn.Module):
    @staticmethod
    def forward(input):
        return torch.sin(input)


class odeNet_NLosc_Mblock(torch.nn.Module):
    def __init__(self, hidden=50, hidden_M=2):
        super(odeNet_NLosc_Mblock, self).__init__()
        self.activation = mySin()
        self.fc1 = torch.nn.Linear(1, hidden)
        self.fc2 = torch.nn.Linear(hidden, hidden)
        self.out = torch.nn.Linear(hidden, hidden_M)
        self.M = torch.nn.Linear(hidden_M, 2)

        self.M.weight.data[0][1] = 0
        self.M.weight.data[1][0] = 0
        self.M.weight.data[0][0] = 1
        self.M.weight.data[1][1] = 1
        self.M.bias.data[0] = 0
        self.M.bias.data[1] = 0
        self.M.weight.requires_grad = False
        self.M.bias.requires_grad = False


    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.out(x)
        x = self.M(x)

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


    def parametric_solution_M(self, t, X_0):
        x = self.fc1(t)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.out(x)

        t_0, x_0, px_0, lam = X_0[0], X_0[1], X_0[2], X_0[3]
        dt = t - t_0
        f = (1 - torch.exp(-dt))
        f = f.squeeze()
        x = x.transpose(0, 1)

        z_1 = x[0] * f + x_0
        z_2 = x[1] * f + px_0

        x = torch.stack((z_1, z_2), dim=0)

        x = x.transpose(0, 1)

        x = self.M(x)

        # Split the output in coordinates and momentum
        x_hat = (x[:, 0]).reshape(-1, 1)
        px_hat = (x[:, 1]).reshape(-1, 1)

        return x_hat, px_hat


def plot_new_idea(init_model, end_model, initial_conditions, end_conditions, t_0, t_final, size, M):
    t = torch.linspace(t_0, t_final, size).reshape(-1, 1)

    plt.figure(figsize=(12, 12))
    plt.title('New Idea\n'
              '({}, {}) ---> ({}, {})'.format(initial_conditions[1], initial_conditions[2],
                                              end_conditions[1], end_conditions[2]))

    # SOURCE
    x, px = init_model.parametric_solution(t, initial_conditions)
    x, px = x.detach().numpy(), px.detach().numpy()
    plt.plot(x, px, 'b', linewidth=2, linestyle='--', label='source')

    # TRANSFORMED
    z = np.hstack((x, px))
    z_hat = np.matmul(M, z.transpose()).transpose()
    x_hat, px_hat = (z_hat[:, 0]).reshape(-1, 1), (z_hat[:, 1]).reshape(-1, 1)
    plt.plot(x_hat, px_hat, 'g', linewidth=2, linestyle='--', label='transformed')

    # TARGET
    x, px = end_model.parametric_solution(t, end_conditions)
    x, px = x.detach().numpy(), px.detach().numpy()
    z = np.hstack((x, px))
    np.save('red_line.npy', z)
    plt.plot(x, px, 'r', linewidth=2, linestyle='--', label='target')

    # X
    plt.plot([x_0], [px_0], 'kx', label='original X0', markersize=14)
    plt.plot([x[0]], [px[0]], 'bx', label='transformed X0', markersize=14)

    plt.ylabel('px')
    plt.xlabel('z')
    plt.xlim(-3.2, 3.2)
    plt.ylim(-3.2, 3.2)
    plt.legend()
    plt.savefig(ROOT_DIR + '/plots/transformed.png')
    plt.show()


def plot_phase(source, M_model, target, x_0, px_0, x_e, px_e, t_0, t_final, size):
    t = torch.linspace(t_0, t_final, size).reshape(-1, 1)
    colors = ['b', 'g', 'r', 'm', 'k', 'y', 'c']
    plt.figure(figsize=(12, 12))
    plt.title('NLosc phase space')
    models = [source, M_model, target]
    names = ['source', 'M', 'target']
    conditions = [[t_0, x_0, px_0, t_final], [t_0, x_e, px_e, t_final], [t_0, x_e, px_e, t_final]]

    # Source
    x, px = models[0].parametric_solution(t, conditions[0])
    x, px = x.detach().numpy(), px.detach().numpy()
    plt.plot(x, px, colors[0], linewidth=2, linestyle='-', label=names[0], alpha=0.9)
    z = np.hstack((x, px))
    np.save('{}.npy'.format(names[0]), z)

    # M
    x, px = models[1].parametric_solution_M(t, conditions[0])
    x, px = x.detach().numpy(), px.detach().numpy()
    plt.plot(x, px, colors[1], linewidth=2, linestyle='-', label=names[1], alpha=0.9)
    z = np.hstack((x, px))
    np.save('{}.npy'.format(names[1]), z)

    # Target
    x, px = models[2].parametric_solution_M(t, conditions[2])
    x, px = x.detach().numpy(), px.detach().numpy()
    plt.plot(x, px, colors[2], linewidth=2, linestyle='-', label=names[2], alpha=0.9)
    z = np.hstack((x, px))
    np.save('{}.npy'.format(names[2]), z)

    plt.ylabel('p')
    plt.xlabel('x')
    plt.xlim(-3.2, 3.2)
    plt.ylim(-3.2, 3.2)
    plt.legend()
    plt.savefig(ROOT_DIR + '/plots/solutions_{}.png')
    plt.show()


def plot_solutions(source, M_model, target, x_0, px_0, x_e, px_e, t_0, t_final, size):
    models = [source, M_model, target]
    names = ['source', 'M', 'target']
    conditions = [[t_0, x_0, px_0, t_final], [t_0, x_e, px_e, t_final], [t_0, x_e, px_e, t_final]]
    colors = ['b', 'g', 'r', 'm', 'k', 'y', 'c']
    t = torch.linspace(t_0, t_final, size).reshape(-1, 1)

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlabel('t', fontsize=18)
    ax1.set_ylabel('x', fontsize=18)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel('t', fontsize=18)
    ax2.set_ylabel('p', fontsize=18)

    # Generate plot for each source_model
    for i in range(len(models)):
        x, px = models[i].parametric_solution(t, conditions[i])
        x, px = x.detach().numpy(), px.detach().numpy()
        t_n = t.detach().numpy()
        ax1.plot(t_n, x, linewidth=2, color=colors[i], label=names[i], alpha=0.7, linestyle='--')
        ax2.plot(t_n, px, linewidth=2, color=colors[i], label=names[i], alpha=0.7, linestyle='--')

    ax1.legend()
    ax2.legend()

    fig.tight_layout()

    plt.savefig('img.png')
    fig.show()


if __name__ == '__main__':
    # SOURCE
    x_0, px_0, lam = 1.0, 1.0, 1
    t_0, t_final, train_size = 0., 4 * np.pi, 200
    initial_conditions = [t_0, x_0, px_0, lam]
    source_model = odeNet_NLosc_Mblock(hidden=50)
    betas = (0.999, 0.9999)
    lr = 8e-4
    optimizer = torch.optim.Adam(source_model.parameters(), lr=lr, betas=betas)

    # Train and save
    # _, losses, _, _ = train_NNLosc(initial_conditions, t_final, lam, hidden=50,
    #                                epochs=int(5e4), train_size=200, optimizer=optimizer, start_model=source_model)
    # torch.save({'model_state_dict': source_model.state_dict()},
    #            ROOT_DIR + '/models/NLosc/M/x_0={}'
    #                       '-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_0, px_0, t_0, t_final, lam))
    # print('Loss pre:', losses[-1])
    # Load
    path = ROOT_DIR + '/models/NLosc/M/x_0={}-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_0, px_0, t_0, t_final, lam)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    source_model.load_state_dict(checkpoint['model_state_dict'])

    # EXPLORER
    n_points = 200
    x_e, px_e, lam = 1.5, 1.5, 1
    t_0, t_final = 0., 4 * np.pi
    end_conditions = [t_0, x_e, px_e, lam]
    explorer_model = odeNet_NLosc_Mblock(hidden=50)
    betas = (0.999, 0.9999)
    lr = 8e-4
    optimizer = torch.optim.Adam(explorer_model.parameters(), lr=lr, betas=betas)

    writer = SummaryWriter('runs/' + 'NLosc_points={}'.format(n_points))
    grid_explorer = torch.linspace(t_0, t_final, n_points).reshape(-1, 1)
    _, losses, _, _ = train_NNLosc(initial_conditions=end_conditions, t_final=t_final, lam=lam,
                                   hidden=50, epochs=int(10e3),
                                   train_size=train_size, optimizer=optimizer, num_batches=1,
                                   start_model=explorer_model, val_size=train_size,
                                   selection=None, perc=1., additional_comment='_explore',
                                   grid=grid_explorer, perturb=False)


    # FINETUNE
    fine_model = copy.deepcopy(source_model)

    fine_model.M.weight.requires_grad = True
    fine_model.M.bias.requires_grad = True
    fine_model.fc1.weight.requires_grad = False
    fine_model.fc1.bias.requires_grad = False
    fine_model.fc2.weight.requires_grad = False
    fine_model.fc2.bias.requires_grad = False
    fine_model.out.weight.requires_grad = False
    fine_model.out.bias.requires_grad = False

    betas = (0.999, 0.9999)
    lr = 8e-4
    train_size = 200
    epochs = int(1e4)
    optimizer = torch.optim.Adam(fine_model.parameters(), lr=lr, betas=betas)

    # Train
    train_NNLosc_Mblock(fine_model, explorer_model, initial_conditions, end_conditions, t_final, lam,
                        int(15e3), train_size, optimizer,
                        grid_explorer=grid_explorer, additional_comment='_Mblock_freeze', gamma=0.8)
    torch.save({'model_state_dict': fine_model.state_dict()},
               ROOT_DIR + '/models/NLosc/M/x_0={}'
                          '-px_0={}-t_0={}-t_f={:.2f}_lam={}_M.pt'.format(x_e, px_e, t_0, t_final, lam))
    # Load
    # fine_model = odeNet_NLosc_Mblock()
    # path = ROOT_DIR + '/models/NLosc/M/x_0={}-px_0={}-t_0={}-t_f={:.2f}_lam={}_M.pt'.format(x_e, px_e, t_0, t_final, lam)
    # checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    # fine_model.load_state_dict(checkpoint['model_state_dict'])

    # Plots
    # Load
    target_model = odeNet_NLosc_Mblock()
    path = ROOT_DIR + '/models/NLosc/M/x_0={}-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_e, px_e, t_0, t_final, lam)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    target_model.load_state_dict(checkpoint['model_state_dict'])

    plot_phase(source_model, fine_model, target_model, x_0, px_0, x_e, px_e, t_0, t_final, train_size)
    plot_solutions(source_model, fine_model, target_model, x_0, px_0, x_e, px_e, t_0, t_final, train_size)

    exit()
    # TARGET MODEL
    target_model = odeNet_NLosc_MM(hidden=50)
    optimizer = torch.optim.Adam(target_model.parameters(), lr=lr, betas=betas)

    # Train and save
    _, losses, _, _ = train_NNLosc(end_conditions, t_final, lam, hidden=50,
                                   epochs=int(5e4), train_size=200, optimizer=optimizer, start_model=target_model)
    torch.save({'model_state_dict': target_model.state_dict()},
               ROOT_DIR + '/models/NLosc/M/x_0={}'
                          '-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_e, px_e, t_0, t_final, lam))
    # # Load
    path = ROOT_DIR + '/models/NLosc/M/x_0={}-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_e, px_e, t_0, t_final, lam)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    target_model.load_state_dict(checkpoint['model_state_dict'])

    x, px = target_model.parametric_solution(grid_explorer, end_conditions)
    x, px = x.detach().numpy(), px.detach().numpy()
    z_true = np.hstack((x, px))


    # MATRIX M
    x, px = source_model.parametric_solution(grid_explorer, initial_conditions)
    x, px = x.detach().numpy(), px.detach().numpy()
    z = np.hstack((x, px))

    x, px = explorer_model.parametric_solution(grid_explorer, end_conditions)
    x, px = x.detach().numpy(), px.detach().numpy()
    z_hat = np.hstack((x, px))

    M = define_M(z, z_true)

    # PLOT
    plot_new_idea(source_model, target_model, initial_conditions, end_conditions, t_0, t_final, train_size, M)
