from models import SIRNetwork
from utils import SIR_solution
from prediction_vs_real import fit_real
from utils import get_known_points
from real_data import *
from training import train_bundle_total
from constants import ROOT_DIR
from torch.utils.tensorboard import SummaryWriter
from shutil import rmtree
from tqdm import tqdm
import copy
from losses import sir_loss
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    t_0 = 0
    t_final = 20

    rescaling = 2

    # Compute the interval in which the equation parameters and the initial conditions should vary
    beta_bundle = [0.5, 0.9]
    gamma_bundle = [0.26, 0.3]
    beta_bundle = [b/rescaling for b in beta_bundle]
    gamma_bundle = [g/rescaling for g in gamma_bundle]
    initial_conditions_set = []
    s_0_bundle = [0.995, 1.0]
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(s_0_bundle)

    # Sanity check
    assert 0 not in beta_bundle and 0 not in gamma_bundle

    # Model parameters
    train_size = 2000
    decay = 0.0
    hack_trivial = False
    epochs = 3000
    lr = 8e-4
    sigma = 0.0

    # Init model
    sir = SIRNetwork(input=4, layers=4, hidden=50)

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/b_s_0={}'
                       '_betas={}_gammas={}_noise_{}.pt'.format(s_0_bundle,
                                                                beta_bundle,
                                                                gamma_bundle, sigma))
    except FileNotFoundError:
        # Train
        optimizer = torch.optim.Adam(sir.parameters(), lr=lr)
        writer = SummaryWriter(
            'runs/' + 'b_s_0={}'
                      '_betas={}_gammas={}_noise_{}.pt'.format(s_0_bundle,
                                                               beta_bundle,
                                                               gamma_bundle, sigma))
        sir, train_losses, run_time, optimizer = train_bundle_total(sir, initial_conditions_set, t_final=t_final,
                                                                    epochs=epochs,
                                                                    num_batches=10, hack_trivial=hack_trivial,
                                                                    train_size=train_size, optimizer=optimizer,
                                                                    decay=decay,
                                                                    writer=writer, betas=beta_bundle,
                                                                    gammas=gamma_bundle)
        # Save the model
        torch.save({'model_state_dict': sir.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIR_bundle_total/b_s_0={}'
                              '_betas={}_gammas={}_noise_{}.pt'.format(s_0_bundle,
                                                                       beta_bundle,
                                                                       gamma_bundle, sigma))

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/b_s_0={}'
                       '_betas={}_gammas={}_noise_{}.pt'.format(s_0_bundle,
                                                                beta_bundle,
                                                                gamma_bundle, sigma))

    # Load the model
    sir.load_state_dict(checkpoint['model_state_dict'])

    # exact_points = get_known_points(model=sir, t_final=t_final, s_0=exact_s_0,
    #                                beta_known=exact_beta, gamma_known=exact_gamma,
    #                                known_size=exact_size)

    writer_dir = 'runs/' + 'real_s_0={}_betas={}_gammas={}.pt'.format(s_0_bundle,
                                                                      beta_bundle,
                                                                      gamma_bundle)

    # Check if the writer directory exists, if yes delete it and overwrite
    if os.path.isdir(writer_dir):
        rmtree(writer_dir)

    writer = SummaryWriter(writer_dir)

    province = lodi
    num_days = 4
    time_unit = 0.25
    time_unit = time_unit * rescaling
    exact_points = get_data_dict(province, num_days=num_days, time_unit=time_unit)
    valid = get_data_dict(province, num_days=None, time_unit=time_unit)

    n_draws = 20
    n_trials = 10
    fit_epochs = 200

    optimal_set = []


    for i in tqdm(range(n_draws), desc='Fitting...'):

        exact_points_tmp = copy.deepcopy(exact_points)

        # Inject some noise in the infected. Noise is gaussian noise with mean 0 and std=sqrt(value)
        for t, v in exact_points_tmp.items():
            noisy_infected = v[1] + abs(np.random.normal(loc=0, scale=np.sqrt(v[1])))
            exact_points_tmp[t] = [1 - noisy_infected, noisy_infected, 0.0]

        min_loss = 1000

        for j in range(n_trials):
            # Search optimal params
            optimal_s_0, optimal_beta, optimal_gamma, rnd_init, traj_mse = fit_real(sir, init_bundle=s_0_bundle,
                                                                                betas=beta_bundle,
                                                                                gammas=gamma_bundle,
                                                                                steps=train_size, lr=1e-3,
                                                                                known_points=exact_points, writer=writer,
                                                                                epochs=fit_epochs)
            if traj_mse <= min_loss:
                optimal_subset = [optimal_s_0, optimal_gamma, optimal_gamma]
                min_loss = traj_mse

        optimal_set.append(optimal_subset)

    overall_infected = []

    # Let's generate the solutions
    for set in optimal_set:
        single_line = []
        single_initial_conditions = [set[0], 1 - set[0], torch.zeros(1, 1)]
        for t, v in exact_points.items():
            t_tensor = torch.Tensor([t]).reshape(-1, 1)
            t_tensor.requires_grad = True

            _, i_single, _ = sir.parametric_solution(t_tensor, single_initial_conditions, beta=set[1],
                                                             gamma=set[2],
                                                             mode='bundle_total')
            single_line.append(i_single.item())
        overall_infected.append(single_line)

    infected_mean = np.mean(overall_infected, axis=0)
    infected_std = np.std(overall_infected, axis=0)

    plt.figure(figsize=(8,5))
    plt.title('Fitting of parameters based on {} days\n'
              'Time unit = {}'.format(num_days, time_unit))
    plt.errorbar(x=range(len(infected_mean)), y=infected_mean, yerr=infected_std)
    plt.show()


