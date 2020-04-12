from models import SIRNetwork
from utils import SIR_solution
from prediction_vs_real import fit_real
from utils import get_known_points
from real_data import *
from training import train_bundle_total
from constants import ROOT_DIR
from torch.utils.tensorboard import SummaryWriter
from shutil import rmtree
from losses import sir_loss
from torch.utils.data import DataLoader
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    t_0 = 0
    t_final = 20

    # Compute the interval in which the equation parameters and the initial conditions should vary
    beta_bundle = [0.5, 0.9]
    gamma_bundle = [0.26, 0.3]
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

    province = cremona
    num_days = 4
    time_unit = 0.25
    exact_points = get_data_dict(province, num_days=num_days, time_unit=time_unit)
    valid = get_data_dict(province, num_days=None, time_unit=time_unit)

    # Search optimal params
    fit_epochs = 500
    optimal_s_0, optimal_beta, optimal_gamma, rnd_init, traj_mse = fit_real(sir, init_bundle=s_0_bundle,
                                                                            betas=beta_bundle,
                                                                            gammas=gamma_bundle,
                                                                            steps=train_size, lr=1e-3,
                                                                            known_points=exact_points, writer=writer,
                                                                            epochs=fit_epochs)

    optimal_initial_conditions = [optimal_s_0, 1 - optimal_s_0, torch.zeros(1, 1)]

    # Let's save the predicted trajectories in the known points
    exact_traj = []  # Solution of the network with the known set of params
    optimal_traj = []  # Solution of the network with the optimal found set of params


    exact_de = 0.
    optimal_de = 0.
    for t, v in exact_points.items():
        t_tensor = torch.Tensor([t]).reshape(-1, 1)
        t_tensor.requires_grad = True

        s_best, i_best, r_best = sir.parametric_solution(t_tensor, optimal_initial_conditions, beta=optimal_beta,
                                                         gamma=optimal_gamma,
                                                         mode='bundle_total')

        optimal_de += sir_loss(t_tensor, s_best, i_best, r_best, optimal_beta, optimal_gamma)

        optimal_traj.append([s_best.item(), i_best.item(), r_best.item()])

    # Exact solution subset
    exact_sub_traj = [exact_points[t] for t in exact_points.keys()]

    exact_mse = 0.
    optimal_mse = 0.
    rnd_mse = 0.
    for idx, p in enumerate(exact_sub_traj):
        exact_s, exact_i, exact_r = p

        exact_mse += (exact_s - exact_traj[idx][0]) ** 2 + (exact_i - exact_traj[idx][1]) ** 2 + (
                exact_r - exact_traj[idx][2]) ** 2
        optimal_mse += (exact_s - optimal_traj[idx][0]) ** 2 + (exact_i - optimal_traj[idx][1]) ** 2 + (
                exact_r - optimal_traj[idx][2]) ** 2

    exact_mse /= len(exact_sub_traj)
    optimal_mse /= len(optimal_traj)

    # Scipy solver solution
    t = np.linspace(0, t_final, t_final)
    s_p, i_p, r_p = SIR_solution(t, optimal_s_0.item(), 1 - optimal_s_0.item(), 0, optimal_beta.item(),
                                 optimal_gamma.item())


    # Generate points between 0 and t_final
    grid = torch.arange(0, t_final, out=torch.FloatTensor()).reshape(-1, 1)
    t_dl = DataLoader(dataset=grid, batch_size=1, shuffle=False)
    s_hat = []
    i_hat = []
    r_hat = []
    for i, t in enumerate(t_dl, 0):
        # Network solutions
        s, i, r = sir.parametric_solution(t, optimal_initial_conditions, beta=optimal_beta, gamma=optimal_gamma,
                                          mode='bundle_total')
        s_hat.append(s.item())
        i_hat.append(i.item())
        r_hat.append(r.item())

    # Plot network solutions
    plt.figure(figsize=(15, 8))
    plt.plot(range(len(s_hat)), s_hat, label='Susceptible - optimal params', color='r', linewidth=2)
    plt.plot(range(len(i_hat)), i_hat, label='Infected - optimal params', color='g', linewidth=2)
    plt.plot(range(len(r_hat)), r_hat, label='Recovered - optimal params', color='b', linewidth=2)
    plt.plot(range(len(s_p)), s_p, label='Susceptible - Scipy', linestyle='--', color='r', linewidth=2)
    plt.plot(range(len(i_p)), i_p, label='Infected - Scipy', linestyle='--', color='g', linewidth=2)
    plt.plot(range(len(r_p)), r_p, label='Recovered - Scipy', linestyle='--', color='b', linewidth=2)
    plt.title('Solving bundle SIR model with found Beta = {} | Gamma = {}\n'
              'Starting found conditions: S0 = {} | I0 = {} | R0 = {:.2f} \n ---\n'
              'Beta bundle = {} | Gamma bundle = {}\n'
              'S0 bundle : {}\n ---\n'
              'Real data available at t: {}\n'
              'Found optimal params: S0 = {} | Beta = {} | Gamma = {}\n'
              'fed into the network yields MSE: {} - DE Loss: {}\n'
              '---\n'
              .format(round(optimal_beta.item(), 4),
                      round(optimal_gamma.item(), 4),
                      round(optimal_s_0.item(), 4),
                      round(1 - optimal_s_0.item(), 4), 0.0,
                      beta_bundle, gamma_bundle, s_0_bundle,
                      list(exact_points.keys()),
                      round(optimal_s_0.item(), 4),
                      round(optimal_beta.item(), 4),
                      round(optimal_gamma.item(), 4), optimal_mse, optimal_de))
    plt.legend(loc='lower right')
    plt.show()

    # Let's compute the MSE over all the known points, not only the subset used for training
    traj_mse = 0.0
    traj_real = []
    traj_hat = []

    # Validation
    for t in valid.keys():
        infected_real = valid[t][1]
        _, infected_hat, _ = sir.parametric_solution(torch.Tensor([t]).reshape(-1, 1), optimal_initial_conditions,
                                                     beta=optimal_beta, gamma=optimal_gamma,
                                                     mode='bundle_total')

        traj_real.append(infected_real)
        traj_hat.append(infected_hat.item())
        traj_mse += (infected_real - infected_hat) ** 2

    traj_mse = traj_mse / len(list(valid.keys()))

    plt.figure(figsize=(8, 5))
    plt.plot(list(valid.keys()), traj_real, label='Real')
    plt.plot(list(valid.keys()), traj_hat, label='Predicted')
    plt.axvline(x=list(exact_points.keys())[-1], color='red', linestyle='--')
    plt.title('Comparison between real infected and predicted infected\n'
              'Number of known points: {}\n'
              'Optimal S(0) = {:.3f} | Optimal Beta = {:.3f} | Optimal Gamma = {:.3f} \n'
              'MSE on training points: {}\n'
              'MSE on all known points: {}'.format(num_days, optimal_s_0.item(), optimal_beta.item(),
                                                   optimal_gamma.item(), optimal_mse,
                                                   traj_mse.item()))
    plt.legend(loc='best')
    plt.show()
