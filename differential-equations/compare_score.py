from training import train_bundle_total
from constants import ROOT_DIR
from torch.utils.tensorboard import SummaryWriter
from models import SIRNetwork
from utils import get_known_points
from shutil import rmtree
from prediction_vs_real import fit_real
from nclmap import nlcmap
import torch
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

if __name__ == '__main__':
    # Initial Conditions
    N = 1
    rescaling_factor = 1

    infected = 0.7
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

    # Compute the interval in which the equation parameters and the initial conditions should vary
    beta_bundle = [0.25, 0.4]
    gamma_bundle = [0.6, 0.75]
    initial_conditions_set = []
    s_0_bundle = [0.5, 0.7]
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(s_0_bundle)

    # Sanity check
    assert i_0 + s_0 + r_0 == rescaling_factor
    assert 0 not in beta_bundle and 0 not in gamma_bundle

    # Model parameters
    train_size = 2000
    decay = 0.0
    hack_trivial = False
    epochs = 3000
    lr = 8e-4

    # Init model
    sir = SIRNetwork(input=4, layers=4, hidden=50)

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/b_s_0={}'
                       '_betas={}_gammas={}_noise_0.pt'.format(s_0_bundle,
                                                       beta_bundle,
                                                       gamma_bundle))
    except FileNotFoundError:
        # Train
        optimizer = torch.optim.Adam(sir.parameters(), lr=lr)
        writer = SummaryWriter(
            'runs/' + 'b_s_0={}_betas={}_gammas={}.pt'.format(s_0_bundle,
                                                                            beta_bundle,
                                                                            gamma_bundle))
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
                              '_betas={}_gammas={}_noise_0.pt'.format(s_0_bundle,
                                                              beta_bundle,
                                                              gamma_bundle))

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/b_s_0={}'
                       '_betas={}_gammas={}_noise_0.pt'.format(s_0_bundle,
                                                       beta_bundle,
                                                       gamma_bundle))

    # Load the model
    sir.load_state_dict(checkpoint['model_state_dict'])

    s_0_draw = np.linspace(s_0_bundle[0], s_0_bundle[1])
    beta_draw = np.linspace(max(0, beta_bundle[0] - 0.35), beta_bundle[1] + 0.35)
    gamma_draw = np.linspace(max(0, gamma_bundle[0] - 0.35), gamma_bundle[1] + 0.35)

    n_draws = 500
    n_trials = 1
    exact_size = 4

    points_score = {}

    for n in tqdm(range(n_draws), desc='Computing scores '):
        rnd_s_0 = np.random.randint(s_0_draw.shape[0], size=1)
        rnd_beta = np.random.randint(beta_draw.shape[0], size=1)
        rnd_gamma = np.random.randint(gamma_draw.shape[0], size=1)

        exact_s_0 = s_0_draw[rnd_s_0]
        exact_beta = beta_draw[rnd_beta]
        exact_gamma = gamma_draw[rnd_gamma]

        exact_points = get_known_points(model=sir, t_final=t_final, s_0=exact_s_0,
                                        beta_known=exact_beta, gamma_known=exact_gamma,
                                        known_size=exact_size)

        min_loss = 1000

        for n in range(n_trials):
            # Search optimal params
            fit_epochs = 100
            optimal_s_0, optimal_beta, optimal_gamma, rnd_init, loss = fit_real(sir, init_bundle=s_0_bundle,
                                                                                betas=beta_bundle,
                                                                                gammas=gamma_bundle,
                                                                                steps=train_size, lr=1e-3,
                                                                                known_points=exact_points, writer=None,
                                                                                epochs=fit_epochs, verbose=False)

            # Compute the score
            score = (optimal_beta.item() - exact_beta) ** 2 + (optimal_gamma.item() - exact_gamma) ** 2 + (
                    optimal_s_0.item() - exact_s_0) ** 2

            if loss < min_loss:
                points_score[(exact_beta.item(), exact_gamma.item())] = score
                min_loss = loss

    points_beta_gamma = list(points_score.keys())
    points_beta_gamma = [list(p) for p in points_beta_gamma]
    points_scores = list(points_score.values())
    points_scores = [s for sublist in points_scores for s in sublist]

    betas = [p[0] for p in points_beta_gamma]
    gammas = [p[1] for p in points_beta_gamma]

    plt.figure(figsize=(10, 10))
    plt.scatter(betas, gammas, c=points_scores, cmap=nlcmap(plt.cm.RdYlGn_r, levels=[1, 2, 10]))
    plt.axvline(beta_bundle[0], linestyle='--')
    plt.axvline(beta_bundle[1], linestyle='--')
    plt.axhline(gamma_bundle[0], linestyle='--')
    plt.axhline(gamma_bundle[1], linestyle='--')
    plt.title('Visualization of score \n'
              'with respect to Beta and Gamma\n'
              'S(0) bundle: {} \n'
              ' Beta bundle: {} | Gamma bundle: {}'.format(s_0_bundle, beta_bundle, gamma_bundle), fontsize=18)
    plt.xlabel('Beta', fontsize=15)
    plt.ylabel('Gamma', fontsize=15)
    plt.colorbar()
    timestamp = datetime.datetime.now().strftime("%H-%M-%S")
    plt.savefig(ROOT_DIR + '\\plots\\' + 'score_bundles-s0={}_beta={}-gamma={}_{}.png'.format(s_0_bundle, beta_bundle,
                                                                                           gamma_bundle, timestamp))
    plt.show()
