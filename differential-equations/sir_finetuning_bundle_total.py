from constants import ROOT_DIR
from models import SIRNetwork
from training import train_bundle_total
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from utils import SIR_solution
import torch
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    source_s_0_bundle = [0.995, 1.0]
    source_beta_bundle = [0.72, 0.88]
    source_gamma_bundle = [0.2, 0.35]
    source_sigma = 0.0

    initial_conditions_set = []
    t_0 = 0
    t_final = 20
    initial_conditions_set.append(t_0)
    initial_conditions_set.append(source_s_0_bundle)

    # Init model
    sir = SIRNetwork(input=4, layers=4, hidden=50)
    lr = 8e-4

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/b_s_0={}'
                       '_betas={}_gammas={}_noise_{}.pt'.format(source_s_0_bundle,
                                                                source_beta_bundle,
                                                                source_gamma_bundle, source_sigma))
    except FileNotFoundError:
        # Train
        optimizer = torch.optim.Adam(sir.parameters(), lr=lr)
        source_epochs = 5000
        source_hack_trivial = False
        source_train_size = 2500
        source_decay = 1e-4
        writer = SummaryWriter(
            'runs/' + 'b_s_0={}_betas={}_gammas={}_noise_{}_scratch.pt'.format(source_s_0_bundle,
                                                                       source_beta_bundle,
                                                                       source_gamma_bundle, source_sigma))
        sir, train_losses, run_time, optimizer = train_bundle_total(sir, initial_conditions_set, t_final=t_final,
                                                                    epochs=source_epochs,
                                                                    num_batches=10, hack_trivial=source_hack_trivial,
                                                                    train_size=source_train_size, optimizer=optimizer,
                                                                    decay=source_decay,
                                                                    sigma=source_sigma,
                                                                    writer=writer, betas=source_beta_bundle,
                                                                    gammas=source_gamma_bundle)
        # Save the model
        torch.save({'model_state_dict': sir.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIR_bundle_total/b_s_0={}'
                              '_betas={}_gammas={}_noise_{}.pt'.format(source_s_0_bundle,

                                                                       source_beta_bundle,
                                                                       source_gamma_bundle, source_sigma))

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/b_s_0={}'
                       '_betas={}_gammas={}_noise_{}.pt'.format(source_s_0_bundle,
                                                                       source_beta_bundle,
                                                                       source_gamma_bundle, source_sigma))

    # Target model




    # Target model
    target_s_0_bundle = [0.995, 1.0]
    target_beta_bundle = [0.5, 0.9]
    target_gamma_bundle = [0.26, 0.30]
    target_sigma = 0.0

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/b_s_0={}'
                       '_betas={}_gammas={}_noise_{}.pt'.format(target_s_0_bundle,
                                                                target_beta_bundle,
                                                                target_gamma_bundle, target_sigma))
    except FileNotFoundError:
        print('Finetuning...')
        # Load old model
        sir.load_state_dict(checkpoint['model_state_dict'])
        # Train
        initial_conditions_set = []
        t_0 = 0
        t_final = 20
        initial_conditions_set.append(t_0)
        initial_conditions_set.append(target_s_0_bundle)
        optimizer = torch.optim.Adam(sir.parameters(), lr=lr)
        target_epochs = 500
        target_hack_trivial = False
        target_train_size = 2500
        target_decay = 1e-4
        writer = SummaryWriter(
            'runs/' + 'b_s_0={}_betas={}_gammas={}_noise_{}_finetuned.pt'.format(target_s_0_bundle,
                                                                       target_beta_bundle,
                                                                       target_gamma_bundle, target_sigma))
        sir, train_losses, run_time, optimizer = train_bundle_total(sir, initial_conditions_set, t_final=t_final,
                                                                    epochs=target_epochs,
                                                                    num_batches=10, hack_trivial=target_hack_trivial,
                                                                    train_size=target_train_size, optimizer=optimizer,
                                                                    decay=target_decay,
                                                                    sigma=target_sigma,
                                                                    writer=writer, betas=target_beta_bundle,
                                                                    gammas=target_gamma_bundle)


        # Save the model
        torch.save({'model_state_dict': sir.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIR_bundle_total/b_s_0={}'
                              '_betas={}_gammas={}_noise_{}.pt'.format(target_s_0_bundle,
                                                                       target_beta_bundle,
                                                                       target_gamma_bundle, target_sigma))

        # Load the checkpoint
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_bundle_total/b_s_0={}'
                       '_betas={}_gammas={}_noise_{}.pt'.format(source_s_0_bundle,
                                                                       source_beta_bundle,
                                                                       source_gamma_bundle, source_sigma))

    # Load fine-tuned model
    sir.load_state_dict(checkpoint['model_state_dict'])

    # Test between 0 and t_final
    grid = torch.arange(0, t_final, out=torch.FloatTensor()).reshape(-1, 1)
    t_dl = DataLoader(dataset=grid, batch_size=1, shuffle=False)
    s_hat = []
    i_hat = []
    r_hat = []

    # Scipy solver solution
    beta_t = 0.8
    gamma_t = 0.3
    s_0 = 0.9988
    i_0 = 1 - s_0
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
        s, i, r = sir.parametric_solution(t, initial_conditions_set, beta=beta_t, gamma=gamma_t, mode='bundle_total')
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
    plt.title('Solving bundle SIR model with Beta = {} | Gamma = {}\n'
              'Starting conditions: S0 = {} | I0 = {} | R0 = {:.2f} \n'
              'Betas = {} | Gammas = {}\n'
              'Starting conditions (s0, i0, r0) : {} | {} | {}\n'
              'Brownian noise - sigma = {}'.format(round(beta_t.item(), 2),
                                                   round(gamma_t.item(), 2),
                                                   s_0, i_0, r_0,
                                                   target_beta_bundle,
                                                   target_gamma_bundle, target_s_0_bundle, i_0, r_0, target_sigma))
    plt.legend(loc='lower right')
    plt.show()
