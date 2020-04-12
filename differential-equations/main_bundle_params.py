from training import train_bundle_params
from constants import ROOT_DIR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
from models import SIR_solution, SIRNetwork

if __name__ == '__main__':

    # Initial Conditions
    N = 1
    rescaling_factor = 1

    infected = 0.3
    susceptible = N - infected
    recovered = 0

    s_0 = susceptible / N * rescaling_factor
    i_0 = infected / N * rescaling_factor
    r_0 = 0

    # Equation parameters
    initial_conditions = [0, [s_0, i_0, r_0]]
    beta = round(0.8, 2)
    gamma = round(0.2, 2)
    beta_std = 0.1 * beta
    gamma_std = 0.1 * gamma
    betas = [round(beta - beta_std, 2), round(beta + beta_std, 2)]
    gammas = [round(gamma - beta_std, 2), round(gamma + gamma_std, 2)]

    # Sanity check
    assert i_0 + s_0 + r_0 == rescaling_factor
    assert 0 not in betas and 0 not in gammas

    # Model parameters
    t_final = 20
    train_size = 2500
    decay = 0.0
    hack_trivial = False
    epochs = 500
    lr = 8e-4

    # Init model
    sir = SIRNetwork(input=3, layers=2, hidden=50)

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_params/b_s_0={:.2f}-i_0={:.2f}-r_0={:.2f}'
                       '_betas={}_gammas={}.pt'.format(s_0,
                                                       i_0, r_0,
                                                       betas,
                                                       gammas))
    except FileNotFoundError:
        # Train
        optimizer = torch.optim.Adam(sir.parameters(), lr=lr)
        writer = SummaryWriter(
            'runs/' + 'b_s_0={:.2f}-i_0={:.2f}-r_0={:.2f}_betas={}_gammas={}.pt'.format(s_0,
                                                                                        i_0, r_0,
                                                                                        betas,
                                                                                        gammas))
        sir, train_losses, run_time, optimizer = train_bundle_params(sir, initial_conditions, t_final=t_final, epochs=epochs,
                                                                     num_batches=10, hack_trivial=hack_trivial,
                                                                     train_size=train_size, optimizer=optimizer, decay=decay,
                                                                     writer=writer, betas=betas, gammas=gammas)
        # Save the model
        torch.save({'model_state_dict': sir.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIR_bundle_params/b_s_0={:.2f}-i_0={:.2f}-r_0={:.2f}'
                              '_betas={}_gammas={}.pt'.format(s_0,
                                                              i_0, r_0,
                                                              betas,
                                                              gammas))

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_params/b_s_0={:.2f}-i_0={:.2f}-r_0={:.2f}'
                       '_betas={}_gammas={}.pt'.format(s_0,
                                                       i_0, r_0,
                                                       betas,
                                                       gammas))

    # Load the model
    sir.load_state_dict(checkpoint['model_state_dict'])

    # Test between 0 and t_final
    grid = torch.arange(0, t_final, out=torch.FloatTensor()).reshape(-1, 1)
    t_dl = DataLoader(dataset=grid, batch_size=1, shuffle=False)
    s_hat = []
    i_hat = []
    r_hat = []

    # Scipy solver solution
    beta_t = 3
    gamma_t = 0.1201
    t = np.linspace(0, t_final, t_final)
    s_p, i_p, r_p = SIR_solution(t, s_0, i_0, r_0, beta_t, gamma_t)

    # Convert beta and gamma to tensor for prediction
    beta_t = torch.Tensor([beta_t]).reshape(-1, 1)
    gamma_t = torch.Tensor([gamma_t]).reshape(-1, 1)

    for i, t in enumerate(t_dl, 0):
        # Network solutions
        s, i, r = sir.parametric_solution(t, initial_conditions, beta=beta_t, gamma=gamma_t, mode='bundle_params')
        s_hat.append(s.item())
        i_hat.append(i.item())
        r_hat.append(r.item())

    # Plot network solutions
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(s_hat)), s_hat, label='Susceptible')
    plt.plot(range(len(i_hat)), i_hat, label='Infected')
    plt.plot(range(len(r_hat)), r_hat, label='Recovered')
    plt.plot(range(len(s_p)), s_p, label='Susceptible - Scipy', linestyle='--')
    plt.plot(range(len(i_p)), i_p, label='Infected - Scipy', linestyle='--')
    plt.plot(range(len(r_p)), r_p, label='Recovered - Scipy', linestyle='--')
    plt.title('Solving bundle SIR model with Betas = {} | Gamma = {} \n'
              'Starting conditions: S0 = {:.2f} | I0 = {:.2f} | R0 = {:.2f} \n'
              'Betas = {} | Gammas = {}'.format(round(beta_t.item(), 2), round(gamma_t.item(), 2), s_0, i_0, r_0, betas,
                                                gammas))
    plt.legend(loc='lower right')
    plt.savefig(
        ROOT_DIR + '/plots/b_SIR_s0={:.2f}_i0={:.2f}_r0={:.2f}_betas={}_gammas={}.png'.format(s_0, i_0, r_0, betas,
                                                                                              gammas))
    plt.show()
