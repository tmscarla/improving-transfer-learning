import torch
import numpy as np
from constants import device
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.integrate import odeint
from scipy.stats import norm
from losses import hamiltonian_eq_loss
from scipy.stats import entropy
from math import isclose, log10


def perturb_points(grid, t_0, t_final, sig=0.5):
    # Stochastic perturbation of the evaluation points
    # Force t[0]=t0 and force points to be in the t-interval
    delta_t = grid[1] - grid[0]
    noise = delta_t * torch.randn_like(grid) * sig
    t = grid + noise
    t.data[2] = torch.ones(1, 1) * (-1)
    t.data[t < t_0] = t_0 - t.data[t < t_0]
    t.data[t > t_final] = 2 * t_final - t.data[t > t_final]
    t.data[0] = torch.ones(1, 1) * t_0
    t.requires_grad = False
    return t


def generate_dataloader(grid, t_0, t_final, batch_size, perturb=True, shuffle=True):
    # Generate a dataloader with perturbed points starting from a grid_explorer
    if perturb:
        grid = perturb_points(grid, t_0, t_final, sig=0.15 * t_final)
    grid.requires_grad = True

    t_dl = DataLoader(dataset=grid, batch_size=batch_size, shuffle=shuffle)
    return t_dl


def save_data(path, t, x, px, E, loss):
    # Save data to disk
    np.savetxt(path + "t.txt", t)
    np.savetxt(path + "x.txt", x)
    np.savetxt(path + "px.txt", px)
    np.savetxt(path + "E.txt", E)
    np.savetxt(path + "Loss.txt", loss)


def compute_losses(points, model, initial_conditions):
    losses = []

    for i, t_i in enumerate(points):
        model.zero_grad()
        t_i = t_i.view(1, 1)
        t_i.requires_grad = True
        t_i, model = t_i.to(device), model.to(device)

        x, px = model.parametric_solution(t_i, initial_conditions)
        loss = hamiltonian_eq_loss(t_i, x, px, 1.0)
        losses.append(loss)
    return losses


def compute_jacobian(points, model, initial_conditions, norm=None):
    jacobians = []

    for i, t_i in enumerate(tqdm(points, desc='Compute Jacobian')):
        model.zero_grad()
        t_i = t_i.view(1, 1)
        t_i.requires_grad = True
        t_i, model = t_i.to(device), model.to(device)

        x, px = model.parametric_solution(t_i, initial_conditions)
        loss = hamiltonian_eq_loss(t_i, x, px, 1.0)
        loss.backward()

        jac = []
        for idx, param in enumerate(model.parameters()):
            jac.append(param.grad.view(-1))
        jac = torch.cat(jac)
        jacobians.append(jac)

    if not norm:
        return jacobians
    elif norm == 'l1':
        return [torch.norm(jac, p=1) for jac in jacobians]
    elif norm == 'l2':
        return [torch.norm(jac, p=2) for jac in jacobians]


def generate_grid(selection, model, initial_conditions, t_0, t_final, size, perc=1):
    perc_size = int(size * perc)
    grid = torch.linspace(t_0, t_final, size).reshape(-1, 1)

    if not selection:
        return grid
    # TODO MAKE JACOBIAN AND LOSS GENERALIZABLE TO DIFFERENT LOSSES
    elif selection == 'jacobian':
        jacobians = compute_jacobian(grid, model, initial_conditions, norm='l2')
        indices = torch.argsort(torch.Tensor(jacobians), descending=True)
        return grid[indices][:perc_size]
    elif selection == 'loss':
        losses = compute_losses(grid, model, initial_conditions)
        indices = torch.argsort(torch.Tensor(losses), descending=True)
        return grid[indices][:perc_size]
    elif selection == 'exponential':
        start = 0
        end = np.log(t_final)
        grid = torch.logspace(start=start, end=end, steps=perc_size, base=np.exp(1))
        grid = grid.unsqueeze(dim=1)
        return grid
    elif selection == 'inv_exponential':
        start = np.log(t_final)
        end = 0
        grid = torch.logspace(start=start, end=end, steps=perc_size, base=np.exp(1))
        grid = grid.unsqueeze(dim=1)
        return grid
    else:
        raise ValueError("Selection must be ['jacobian', 'loss']")


# Use below in the Scipy Solver
def f(u, t, beta, gamma):
    s, i, r = u  # unpack current values of u
    N = s + i + r
    derivs = [-(beta * i * s) / N, (beta * i * s) / N - gamma * i, gamma * i]  # list of dy/dt=f functions
    return derivs


# Scipy Solver
def SIR_solution(t, s_0, i_0, r_0, beta, gamma):
    u_0 = [s_0, i_0, r_0]

    # Call the ODE solver
    sol_sir = odeint(f, u_0, t, args=(beta, gamma))
    s = sol_sir[:, 0]
    i = sol_sir[:, 1]
    r = sol_sir[:, 2]

    return s, i, r


if __name__ == '__main__':
    from constants import ROOT_DIR
    import matplotlib.pyplot as plt

    t_final = 20
    N = 1
    rescaling_factor = 1

    infected = 0.2
    susceptible = N - infected
    recovered = 0

    s_0 = susceptible / N * rescaling_factor
    i_0 = infected / N * rescaling_factor
    r_0 = 0
    beta = 0.8
    gamma = 0.2
    t = np.linspace(0, t_final, t_final)
    s_p, i_p, r_p = SIR_solution(t, s_0, i_0, r_0, beta, gamma)

    ### Plot Scipy solutions ###
    plt.figure(figsize=(15, 5))
    plt.plot(range(len(s_p)), s_p, label='Susceptible - Scipy')
    plt.plot(range(len(i_p)), i_p, label='Infected - Scipy')
    plt.plot(range(len(r_p)), r_p, label='Recovered - Scipy')
    plt.title('Scipy Solver SIR model with Beta = {} | Gamma = {} \n'
              'Starting conditions: S0 = {:.2f} | I0 = {:.2f} | R0 = {:.2f} \n'.format(beta, gamma, s_0, i_0, r_0))
    plt.legend(loc='lower right')
    plt.show()


def modularize_decay(model, t_final, initial_conditions, decay, steps=5):
    # Select some samples from the timespan
    grid = torch.linspace(0, t_final, steps).reshape(-1, 1)
    t_dl = DataLoader(dataset=grid, batch_size=1, shuffle=False)

    # Compute their prediction
    s_check, i_check, r_check = [], [], []
    for i, t in enumerate(t_dl, 0):
        s, i, r = model.parametric_solution(t, initial_conditions)
        s_check.append(s.item())
        i_check.append(i.item())
        r_check.append(r.item())

    # TODO FIX THIS!
    # If the predictions are not entropic, I increase the decay
    max_ent = entropy(steps * [1 / steps])
    i_ent = entropy(i_check)
    if not isclose(i_ent, max_ent, abs_tol=0.1):
        decay = decay * 1.1

    return decay


def get_known_points(model, t_final, s_0, beta_known, gamma_known, known_size):
    model.eval()

    # Generate tensors and get known points from the ground truth
    grid = torch.linspace(0, t_final, t_final).reshape(-1, 1)
    exact_initial_conditions = [torch.Tensor([s_0]).reshape(-1, 1), torch.Tensor([1 - s_0]).reshape(-1, 1)]
    beta_known = torch.Tensor([beta_known]).reshape(-1, 1)
    gamma_known = torch.Tensor([gamma_known]).reshape(-1, 1)

    exact_traj = []

    for t in grid:
        t = t.reshape(-1, 1)
        s_p, i_p, r_p = model.parametric_solution(t, exact_initial_conditions,
                                              beta=beta_known,
                                              gamma=gamma_known,
                                              mode='bundle_total')
        exact_traj.append([s_p.item(), i_p.item(), r_p.item()])

    rnd_t = np.linspace(0, int(len(exact_traj)) - 1, known_size)

    known_points = {}

    for t in rnd_t:
        known_points[t] = [exact_traj[int(t)][0], exact_traj[int(t)][1], exact_traj[int(t)][2]]

    return known_points


def generate_brownian(t_0, t_final, N, sigma):
    dt = (t_final - t_0) / N

    m = abs(int(log10(dt)))

    noise = {}

    for i in range(N + 1):
        if i == 0:
            noise[0] = 0
        else:
            noise[round(t_0 + i * dt, m)] = noise[round(t_0 + (i - 1) * dt, m)] + np.sqrt(sigma * dt) * norm.rvs()


    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(list(noise.values()))
    print('Noise avg: {:.3f}'.format(np.mean(list(noise.values()))))
    plt.show()

    return noise


