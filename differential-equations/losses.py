import torch
import numpy as np
from torch.autograd import grad
from constants import device, dtype


def sir_loss(t, s, i, r, beta, gamma, noise=None, sigma=0, decay=0):
    s_prime = dfx(t, s)
    i_prime = dfx(t, i)
    r_prime = dfx(t, r)

    # If noise, I fetch the amount of noise for each time step, else noise is just a zero tensor
    if noise:
        sub_noise = []
        for time_step in t:
            # Round up to get the noise_value
            time_key = round(time_step.item(), 2)
            sub_noise.append(noise[time_key])
        sub_noise = torch.Tensor(sub_noise).reshape(-1, 1)
    else:
        sub_noise = torch.zeros((t.shape[0], 1))


    #N = s + i + r
    N = 1

    noise_term = (sigma * sub_noise * s * i) / N

    loss_s = s_prime + (beta * i * s) / N + noise_term
    loss_i = i_prime - (beta * i * s) / N + gamma * i - noise_term
    loss_r = r_prime - gamma * i

    # Regularize to give more importance to initial points
    loss_s = loss_s * torch.exp(-decay * t)
    loss_i = loss_i * torch.exp(-decay * t)
    loss_r = loss_r * torch.exp(-decay * t)

    loss_s = (loss_s.pow(2)).mean()
    loss_i = (loss_i.pow(2)).mean()
    loss_r = (loss_r.pow(2)).mean()

    total_loss = loss_s + loss_i + loss_r

    return total_loss


def mse_loss(known, model, initial_conditions):
    mse_loss = 0.
    for t in known.keys():
        t_tensor = torch.Tensor([t]).reshape(-1, 1)
        s_hat, i_hat, r_hat = model.parametric_solution(t_tensor, initial_conditions)
        loss_s = (known[t][0] - s_hat).pow(2)
        loss_i = (known[t][1] - i_hat).pow(2)
        loss_r = (known[t][2] - i_hat).pow(2)

        mse_loss += loss_s + loss_i + loss_r
    return mse_loss


def trivial_loss(model, t_final, initial_conditions, method):
    if method == 'mse':
        mse_loss = 0.

        grid = torch.linspace(t_final * 0.25, t_final * 0.75, steps=50)

        for t in grid:
            t = t.reshape(-1, 1)
            s_hat, _, _ = model.parametric_solution(t, initial_conditions)
            mse_loss += (1.0 - s_hat) ** 2

        return 1 / mse_loss

    elif method == 'derivative':
        grid = torch.linspace(0, t_final, steps=20)
        grid.requires_grad = True

        derivative_loss = 0
        for t in grid:
            t = t.reshape(-1, 1)
            s_hat, i_hat, r_hat = model.parametric_solution(t, initial_conditions)
            s_prime = dfx(t, s_hat)
            i_prime = dfx(t, i_hat)
            r_prime = dfx(t, r_hat)

            derivative_loss += s_prime ** (-1) + i_prime ** (-1) + r_prime ** (-1)

        return 0.001 * derivative_loss


def dfx(x, f):
    # Calculate the derivative with auto-differentiation
    x = x.to(device)
    grad_outputs = torch.ones(x.shape, dtype=dtype)
    grad_outputs = grad_outputs.to(device)

    return grad([f], [x], grad_outputs=grad_outputs, create_graph=True)[0]


def hamiltonian_eq_loss(t, x, px, lam):
    # Define the loss function by Hamilton Eqs., write explicitly the Ham. Equations
    x_d, pxd = dfx(t, x), dfx(t, px)
    fx = x_d - px
    fpx = pxd + x + lam * x.pow(3)
    loss_x = (fx.pow(2)).mean()
    loss_px = (fpx.pow(2)).mean()
    total_loss = loss_x + loss_px
    return total_loss


def hamiltonian_eq_loss_by_h(t, x, px, lam):
    # This is an alternative way to define the loss function:
    # Define the loss function by Hamilton Eqs. directly from Hamiltonian H

    # Potential and Kinetic Energy
    V = 0.5 * x.pow(2) + lam * x.pow(4) / 4
    K = 0.5 * px.pow(2)
    ham = K + V
    xd, pxd = dfx(t, x), dfx(t, px)

    # Calculate the partial spatial derivatives of H
    h_x = dfx(ham, x)
    h_px = dfx(ham, px)

    # Hamilton Eqs
    fx = xd - h_px
    fpx = pxd + h_x
    loss_x = (fx.pow(2)).mean()
    loss_px = (fpx.pow(2)).mean()
    total_loss = loss_x + loss_px
    return total_loss


def mblock_loss(x, p, x_true, p_true):
    fx = x_true - x
    fp = p_true - p
    loss_x = (fx.pow(2)).mean()
    loss_p = (fp.pow(2)).mean()

    total_loss = loss_x + loss_p
    return total_loss
