from training import train_bundle_init
from constants import ROOT_DIR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
from models import SIRNetwork
from utils import SIR_solution

if __name__ == '__main__':

    # Initial Conditions
    N = 1
    rescaling_factor = 1

    infected = 0.22
    susceptible = N - infected
    recovered = 0

    s_0 = susceptible / N * rescaling_factor
    i_0 = infected / N * rescaling_factor
    r_0 = 0

    # Equation parameters
    beta = round(0.8, 2)
    gamma = round(0.2, 2)
    t_0 = 0
    t_final = 20

    # Stochasticity effect
    sigma = 0

    # Compute the interval in which the initial conditions should vary
    initial_conditions_set = []
    s_0_set = [0.7, 0.9]
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(s_0_set)

    # Sanity check
    assert i_0 + s_0 + r_0 == rescaling_factor

    # Model parameters
    train_size = 2000
    decay = 0.0001
    hack_trivial = False
    epochs = 1000
    lr = 8e-4

    # Init model
    sir = SIRNetwork(input=2, layers=2, hidden=50)

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_init/s_0={}-i_0={}-r_0={}'
                       '_beta={}_gamma={}_noise_{}.pt'.format(s_0_set,
                                                              i_0, r_0,
                                                              beta,
                                                              gamma, sigma))
    except FileNotFoundError:
        # Train
        optimizer = torch.optim.Adam(sir.parameters(), lr=lr)
        writer = SummaryWriter(
            'runs/' + 's_0={}-i_0={}-r_0={}_beta={}_gamma={}_noise_{}.pt'.format(s_0_set,
                                                                                 i_0, r_0,
                                                                                 beta,
                                                                                 gamma, sigma))
        sir, train_losses, run_time, optimizer = train_bundle_init(sir, initial_conditions_set, t_final=t_final,
                                                                   epochs=epochs,
                                                                   num_batches=10, hack_trivial=hack_trivial,
                                                                   train_size=train_size, optimizer=optimizer,
                                                                   decay=decay,
                                                                   sigma=sigma,
                                                                   writer=writer, beta=beta, gamma=gamma)
        # Save the model
        torch.save({'model_state_dict': sir.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIR_bundle_init/s_0={}-i_0={}-r_0={}'
                              '_beta={}_gamma={}_noise_{}.pt'.format(s_0_set,
                                                                     i_0, r_0,
                                                                     beta,
                                                                     gamma, sigma))

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_init/s_0={}-i_0={}-r_0={}'
                       '_beta={}_gamma={}_noise_{}.pt'.format(s_0_set,
                                                              i_0, r_0,
                                                              beta,
                                                              gamma, sigma))

    # Load the model
    sir.load_state_dict(checkpoint['model_state_dict'])

    # Test between 0 and t_final
    grid = torch.arange(0, t_final, out=torch.FloatTensor()).reshape(-1, 1)
    t_dl = DataLoader(dataset=grid, batch_size=1, shuffle=False)
    s_hat = []
    i_hat = []
    r_hat = []

    # Scipy solver solution
    beta_t = 0.8
    gamma_t = 0.2
    s_0 = 0.8
    i_0 = round(1 - s_0, 2)
    r_0 = 0.0
    t = np.linspace(0, t_final, t_final)
    s_p, i_p, r_p = SIR_solution(t, s_0, i_0, r_0, beta_t, gamma_t)

    # Convert initial conditions, beta and gamma to tensor for prediction
    beta_t = torch.Tensor([beta_t]).reshape(-1, 1)
    gamma_t = torch.Tensor([gamma_t]).reshape(-1, 1)
    s_0_t = torch.Tensor([s_0]).reshape(-1, 1)
    i_0_t = torch.Tensor([i_0]).reshape(-1, 1)
    r_0_t = torch.Tensor([r_0]).reshape(-1, 1)
    initial_conditions_set = [s_0_t, i_0_t, r_0_t]

    for i, t in enumerate(t_dl, 0):
        # Network solutions
        s, i, r = sir.parametric_solution(t, initial_conditions_set, beta=beta_t, gamma=gamma_t, mode='bundle_init')
        s_hat.append(s.item())
        i_hat.append(i.item())
        r_hat.append(r.item())

    blue = '#3366ff'
    red = '#cc0000'
    green = '#13842e'

    # Plot network solutions
    plt.figure(figsize=(12, 5))
    linewidth = 1.5
    plt.plot(range(len(s_hat)), s_hat, label='Susceptible', color=blue, linewidth=linewidth)
    plt.plot(range(len(i_hat)), i_hat, label='Infected', color=red, linewidth=linewidth)
    plt.plot(range(len(r_hat)), r_hat, label='Recovered', color=green, linewidth=linewidth)
    plt.plot(range(len(s_p)), s_p, label='Susceptible - Scipy', linestyle='--', color=blue, linewidth=linewidth)
    plt.plot(range(len(i_p)), i_p, label='Infected - Scipy', linestyle='--', color=red, linewidth=linewidth)
    plt.plot(range(len(r_p)), r_p, label='Recovered - Scipy', linestyle='--', color=green, linewidth=linewidth)
    plt.title('Solving SIR model with Beta = {} | Gamma = {}\n'
              'Starting conditions: S0 = {} | I0 = {} | R0 = {:.2f} \n'
              'Model trained with S(0) in bundle {}\n'.format(round(beta_t.item(), 2),
                                                              round(gamma_t.item(), 2),
                                                              s_0, i_0, r_0,
                                                              s_0_set))
    plt.legend(loc='lower right')
    plt.savefig(ROOT_DIR + '/plots/s_0={}-i_0={}-r_0={}'
                       '_beta={}_gamma={}_noise_{}.png'.format(s_0_set,
                                                              i_0, r_0,
                                                              beta,
                                                              gamma, sigma))
    plt.xlabel('Time')
    plt.ylabel('S(t), I(t), R(t)')
    plt.show()

