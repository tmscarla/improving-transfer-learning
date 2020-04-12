from tqdm import tqdm
from random import shuffle
from losses import sir_loss
import torch
import copy
import numpy as np


def fit_real(model, init_bundle, betas, gammas, known_points, steps, writer, epochs=100, lr=8e-4, verbose=False):
    model.eval()

    betas = torch.linspace(betas[0], betas[1], steps=steps).reshape(-1, 1)
    gammas = torch.linspace(gammas[0], gammas[1], steps=steps).reshape(-1, 1)
    init_bundle = torch.linspace(init_bundle[0],
                                 init_bundle[1], steps=steps).reshape(-1, 1)

    # Sample randomly initial conditions, beta and gamma
    rnd_beta = np.random.randint(betas.shape[0], size=1)
    rnd_gamma = np.random.randint(gammas.shape[0], size=1)
    rnd_init_s_0 = np.random.randint(init_bundle.shape[0], size=1)
    beta = betas[rnd_beta]
    gamma = gammas[rnd_gamma]
    s_0 = init_bundle[rnd_init_s_0].reshape(-1, 1)
    i_0 = (1 - s_0).reshape(-1, 1)  # Set i_0 to be 1-s_0 to enforce that the sum of the initial conditions is 1
    r_0 = torch.zeros(1, 1)  # We fix recovered people at day zero to zero

    rnd_init = [round(s_0.item(), 5), round(beta.item(), 5), round(gamma.item(), 5)]

    # Set requires_grad = True to the inputs to allow backprop
    s_0.requires_grad = True
    beta.requires_grad = True
    gamma.requires_grad = True

    initial_conditions = [s_0, i_0, r_0]

    # Init the optimizer
    optimizer = torch.optim.Adam([s_0, beta, gamma], lr=lr)


    known_t = copy.deepcopy(list(known_points.keys()))

    losses = []

    # Iterate for epochs to find best initial conditions, beta, and gamma that optimizes the difference between
    # my prediction and the real data
    for epoch in tqdm(range(epochs), desc='Finding the best inputs',disable=not verbose):
        optimizer.zero_grad()

        mse_loss = 0.

        # Take the time points and shuffle them
        shuffle(known_t)

        for t in known_t:
            v = known_points[t]

            # Add some random noise
            #t_n = t + np.random.normal(0, 0.001, 1)
            t_n = t + np.random.normal(0, 0.00, 1)

            t_tensor = torch.Tensor([t_n]).reshape(-1, 1)

            s_hat, i_hat, r_hat = model.parametric_solution(t_tensor, initial_conditions, beta=beta, gamma=gamma,
                                                            mode='bundle_total')
            mse_loss_s = (v[0] - s_hat).pow(2)
            mse_loss_i = (v[1] - i_hat).pow(2)
            mse_loss_r = (v[2] - r_hat).pow(2)

            mse_loss += mse_loss_s + mse_loss_i + mse_loss_r

        mse_loss = mse_loss / len(known_points.keys())
        losses.append(mse_loss)

        mse_loss.backward()
        optimizer.step()

        if writer:
            writer.add_scalar('Loss/train', mse_loss, epoch)


    return s_0, beta, gamma, rnd_init, losses[-1]
