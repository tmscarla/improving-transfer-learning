from training import train
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

    infected = 0.2
    susceptible = N - infected
    recovered = 0

    s_0 = susceptible / N * rescaling_factor
    i_0 = infected / N * rescaling_factor
    r_0 = 0

    # Equation parameters
    initial_conditions = [0, [s_0, i_0, r_0]]
    beta = round(0.8, 2)
    gamma = round(0.2, 2)

    # Sanity check
    assert i_0 + s_0 + r_0 == rescaling_factor

    # Model parameters
    t_final = 20
    train_size = 2500
    decay = 0.0
    hack_trivial = False
    epochs = 1000
    lr = 8e-4

    # Scipy solver solution
    t = np.linspace(0, t_final, t_final)
    s_p, i_p, r_p = SIR_solution(t, s_0, i_0, r_0, beta, gamma)

    # Init model
    sir = SIRNetwork(layers=2, hidden=50)

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR/s_0={:.2f}-i_0={:.2f}-r_0={:.2f}'
                       '-t_0={}-t_f={:.2f}_beta={}_gamma={}.pt'.format(s_0,
                                                                       i_0, r_0,
                                                                       initial_conditions[0],
                                                                       t_final, beta,
                                                                       gamma))
    except FileNotFoundError:
        # Train
        optimizer = torch.optim.Adam(sir.parameters(), lr=lr)
        writer = SummaryWriter(
            'runs/' + 's_0={:.2f}-i_0={:.2f}-r_0={:.2f}-t_0={:.2f}-t_f={:.2f}_beta={}_gamma={}.pt'.format(s_0,
                                                                                                          i_0, r_0,
                                                                                                          initial_conditions[
                                                                                                              0],
                                                                                                          t_final, beta,
                                                                                                          gamma))
        sir, train_losses, run_time, optimizer = train(sir, initial_conditions, t_final=t_final, epochs=epochs,
                                                       num_batches=10, hack_trivial=hack_trivial,
                                                       train_size=train_size, optimizer=optimizer,
                                                       decay=decay,
                                                       writer=writer, beta=beta, gamma=gamma)
        # Save the model
        torch.save({'model_state_dict': sir.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIR/s_0={:.2f}-i_0={:.2f}-r_0={:.2f}'
                              '-t_0={}-t_f={:.2f}_beta={}_gamma={}.pt'.format(s_0,
                                                                              i_0, r_0,
                                                                              initial_conditions[0],
                                                                              t_final,
                                                                              beta,
                                                                              gamma))
        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR/s_0={:.2f}-i_0={:.2f}-r_0={:.2f}'
                       '-t_0={}-t_f={:.2f}_beta={}_gamma={}.pt'.format(s_0,
                                                                       i_0, r_0,
                                                                       initial_conditions[0],
                                                                       t_final, beta,
                                                                       gamma))

    # Load the model
    sir.load_state_dict(checkpoint['model_state_dict'])

    # Test between 0 and t_final
    grid = torch.arange(0, t_final, out=torch.FloatTensor()).reshape(-1, 1)
    t_dl = DataLoader(dataset=grid, batch_size=1, shuffle=False)
    s_hat = []
    i_hat = []
    r_hat = []

    for i, t in enumerate(t_dl, 0):
        # Network solutions
        s, i, r = sir.parametric_solution(t, initial_conditions)
        s_hat.append(s.item())
        i_hat.append(i.item())
        r_hat.append(r.item())

    # Colors and Linewidth
    blue = '#3366ff'
    red = '#cc0000'
    green = '#13842e'
    linewidth = 1.5
    # Plot network solutions
    plt.figure(figsize=(12, 5))
    plt.plot(range(len(s_hat)), s_hat, label='Susceptible', color=blue, linewidth=linewidth)
    plt.plot(range(len(i_hat)), i_hat, label='Infected', color=red, linewidth=linewidth)
    plt.plot(range(len(r_hat)), r_hat, label='Recovered', color=green, linewidth=linewidth)
    plt.plot(range(len(s_p)), s_p, label='Susceptible - Scipy', linestyle='--', color=blue, linewidth=linewidth)
    plt.plot(range(len(i_p)), i_p, label='Infected - Scipy', linestyle='--', color=red, linewidth=linewidth)
    plt.plot(range(len(r_p)), r_p, label='Recovered - Scipy', linestyle='--', color=green, linewidth=linewidth)
    plt.title('Solving SIR model with Beta = {} | Gamma = {} \n'
              'Starting conditions: S(0) = {:.2f} | I(0) = {:.2f} | R(0) = {:.2f} \n'.format(beta, gamma, s_0, i_0, r_0))
    plt.legend(loc='lower right')
    plt.xlabel('Time')
    plt.ylabel('S(t), I(t), R(t)')
    plt.savefig(
        ROOT_DIR + '/plots/SIR_s0={:.2f}_i0={:.2f}_r0={:.2f}_beta={}_gamma={}.png'.format(s_0, i_0, r_0, beta, gamma))
    plt.show()

    # Compute loss as a function of the time
    log_losses = []
    for i, t in enumerate(t_dl, 0):
        from losses import sir_loss

        t.requires_grad = True
        s, i, r = sir.parametric_solution(t, initial_conditions)
        t_loss = sir_loss(t, s, i, r, beta, gamma)
        log_losses.append(np.log(t_loss.item()))

    plt.figure(figsize=(15, 5))
    plt.plot(range(len(log_losses)), log_losses)
    plt.xlabel('Time')
    plt.ylabel('Logloss')
    plt.title('Solving SIR model with Beta = {} | Gamma = {} \n'
              'Starting conditions: S(0) = {:.2f} | I(0) = {:.2f} | R(0) = {:.2f} \n'.format(beta, gamma, s_0, i_0, r_0))
    plt.show()
