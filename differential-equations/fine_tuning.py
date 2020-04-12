import numpy as np
from constants import ROOT_DIR
from training_NNLosc import train_NNLosc
from models import odeNet_NLosc_MM, odeNet_NLosc_modular
from losses import dfx
import torch


def find_tau(model, initial_conditions, end_conditions, t_final, epochs, lr, return_values=False):
    t_0, x_0, px_0, lam = initial_conditions
    t_0, x_e, px_e, lam = end_conditions

    # Generate a random value in the interval [t_0, t_final)
    tau = t_0 + torch.rand(1) * (t_final - t_0)
    tau = tau.reshape(-1, 1)
    tau.requires_grad = True

    # Find tau that minimize distance between learnt trajectory and (x_e, px_e)
    for e in range(epochs):
        x_hat, px_hat = model.parametric_solution(tau, initial_conditions)
        distance = torch.sqrt((x_e-x_hat)**2 + (px_e-px_hat)**2)

        dt = dfx(tau, distance)
        tau = tau - (lr * dt)

    # Convert from tensor to float
    tau = tau.detach().numpy()[0][0]

    if return_values:
        return x_hat, px_hat
    else:
        return tau


def train_NNLosc_(x_0=1.3, px_0=1., lam=1, t_0=0., t_final=4 * np.pi, train_size=200,
                  lr=8e-4, hidden=50, epochs=int(4e4), num_batches=1, save=True):
    initial_conditions = [t_0, x_0, px_0, lam]
    betas = (0.999, 0.9999)
    model = odeNet_NLosc_MM(hidden=hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=betas)
    # Use just one batch for the training. No significant different in using more than one.
    model, loss, runtime, optimizer = train_NNLosc(initial_conditions=initial_conditions, t_final=t_final, lam=lam,
                                                   hidden=hidden, epochs=epochs, train_size=train_size,
                                                   optimizer=optimizer, num_batches=num_batches, start_model=model)

    if save:
        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/NLosc/x_0={}'
                              '-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_0, px_0, t_0, t_final, lam))
    return model, loss, runtime, optimizer


if __name__ == '__main__':
    # x_0, px_0, lam = 0.0, 2.0, 0
    # t_0, t_final, train_size = 0., 4 * np.pi, 200
    # initial_conditions = [t_0, x_0, px_0, lam]
    # path = ROOT_DIR + '/models/NLosc/x_0={}-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_0, px_0, t_0, t_final, lam)
    # checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    # source_model = odeNet_NLosc_MM(hidden=50)
    # source_model.load_state_dict(checkpoint['model_state_dict'])
    # from generate_plots import plot_trajectory_NLosc
    # plot_trajectory_NLosc(source_model, initial_conditions, t_0, t_final, train_size)
    # exit()

    x_0 = 1.5; x_e = 2.0
    px_0 = 1.5; px_e = 2.0
    lam = 1
    t_0 = 0.0
    t_final = 4 * np.pi
    # _, losses, _, _ = train_NNLosc_(x_0, px_0, lam, epochs=int(5e4))
    # exit()

    path = ROOT_DIR + '/models/NLosc/x_0={}-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_0, px_0, t_0, t_final, lam)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    source_model = odeNet_NLosc_MM(hidden=50)
    source_model.load_state_dict(checkpoint['model_state_dict'])

    path = ROOT_DIR + '/models/NLosc/x_0={}-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_e, px_e, t_0, t_final, lam)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    fine_model = odeNet_NLosc_MM(hidden=50)
    fine_model.load_state_dict(checkpoint['model_state_dict'])

    from generate_plots import plot_solutions
    plot_solutions(source_model, fine_model, x_0, px_0, x_e, px_e, t_0, t_final, lam, size=200)
    exit()

    model, _, _, optimizer = train_NNLosc()
    model = odeNet_NLosc_MM(hidden=50)
    optimizer = torch.optim.Adam(model.parameters())

    path = ROOT_DIR + '/models/NLosc/x_0=1.3-px_0=1.0-t_0=0.0-t_f=12.57.pt'

    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    torch.optim.Adam(model.parameters())
