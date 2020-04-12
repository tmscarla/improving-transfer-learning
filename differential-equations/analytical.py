import numpy as np
import torch
from constants import *
from playground import odeNet_NLosc_STR
import matplotlib.pyplot as plt


def define_M(z, z_hat):
    inv = np.linalg.inv(np.matmul(z.transpose(), z))
    partial = np.matmul(inv, z.transpose())
    M = np.matmul(partial, z_hat)

    return M


def define_A(x_0, px_0, x_e, px_e, rotation_only=False, pavlos=False):
    if pavlos:
        Q = (px_e * x_0 - x_e * px_0) / (x_e * x_0 + px_0 * px_e)
        theta = np.arctan(Q)
        #theta = theta * 180 / np.pi
        gamma = x_e / (np.cos(theta) * x_0 - np.sin(theta) * x_e)
    else:
        gamma = np.sqrt((x_e**2 + px_e**2) / (x_0**2 + px_0**2))
        cos_theta = (x_0*x_e + px_0*px_e) / (gamma*(x_0**2 + px_0**2))
        theta = np.arccos(cos_theta)

    if rotation_only:
        A = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]], dtype=np.float32)
    else:
        A = np.array([[gamma*np.cos(theta), -gamma*np.sin(theta)],
                      [gamma*np.sin(theta), gamma*np.cos(theta)]], dtype=np.float32)

    return A


def get_gamma_and_theta(A):
    gamma = np.sqrt(np.power(A[0][0], 2) + A[0][1]**2)
    theta = np.arccos(A[0][0] / gamma)
    theta = theta * 180 / np.pi

    return gamma, theta


def transform(z, A):
    A = torch.from_numpy(A)
    z_T = z.transpose(0, 1)
    z_hat = torch.matmul(A, z_T)
    return z_hat.transpose(0, 1)


def transform_one_by_one(x, px, A):
    x_hat, px_hat = [], []

    for i in range(len(x)):
        z_i = np.array([x[i], px[i]])
        z_i.shape = (2, 1)
        z_i_hat = np.matmul(A, z_i)
        x_hat.append(z_i_hat[0][0])
        px_hat.append(z_i_hat[1][0])

    x_hat = np.array(x_hat, dtype=np.float32)
    x_hat.shape = (len(x_hat), 1)
    px_hat = np.array(px_hat, dtype=np.float32)
    px_hat.shape = (len(px_hat), 1)

    return x_hat, px_hat





def plot_analytical_NLosc(init_model, end_model, initial_conditions, end_conditions, t_0, t_final, size):
    t = torch.linspace(t_0, t_final, size).reshape(-1, 1)

    plt.figure(figsize=(12, 12))
    plt.title('Analytical solution\n'
              '({}, {}) ---> ({}, {})'.format(initial_conditions[1], initial_conditions[2],
                                              end_conditions[1], end_conditions[2]))

    x, px = init_model.parametric_solution(t, initial_conditions)
    x, px = x.detach().numpy(), px.detach().numpy()

    array = np.hstack((x, px))
    np.save('blue_line.npy', array)

    plt.plot(x, px, 'b', linewidth=2, linestyle='--', label='init')

    A = define_A(initial_conditions[1], initial_conditions[2], end_conditions[1], end_conditions[2])
    x_a, px_a = transform_one_by_one(x, px, A)
    plt.plot(x_a, px_a, 'g', linewidth=2, linestyle='--', label='tommaso', alpha=0.8)

    A = define_A(initial_conditions[1], initial_conditions[2], end_conditions[1], end_conditions[2], pavlos=True)
    x_a, px_a = transform_one_by_one(x, px, A)
    plt.plot(x_a, px_a, 'orange', linewidth=2, linestyle='dotted', label='pavlos', alpha=0.8)

    x, px = end_model.parametric_solution(t, end_conditions)
    x, px = x.detach().numpy(), px.detach().numpy()
    plt.plot(x, px, 'r', linewidth=2, linestyle='--', label='end')

    plt.plot([x_0], [px_0], 'kx', label='original X0', markersize=14)
    plt.plot([x_a[0]], [px_a[0]], 'bx', label='transformed X0', markersize=14)

    plt.ylabel('px')
    plt.xlabel('z')
    plt.xlim(-3.2, 3.2)
    plt.ylim(-3.2, 3.2)
    plt.legend()
    plt.savefig(ROOT_DIR + '/plots/analytical_solution.png')
    plt.show()


if __name__ == '__main__':
    # Params
    x_0, px_0, lam = 1.0, 1.0, 1
    t_0, t_final, size = 0., 4 * np.pi, 200
    initial_conditions = [t_0, x_0, px_0, lam]
    x_e, px_e, lam = 2.0, 2.0, 1
    end_conditions = [t_0, x_e, px_e, lam]

    # Load pre-trained source_model
    init_model = odeNet_NLosc_STR(hidden=50)
    path = ROOT_DIR + '/models/NLosc/STR/x_0={}-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_0, px_0, t_0, t_final, lam)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    init_model.load_state_dict(checkpoint['model_state_dict'])

    # Load end source_model
    end_model = odeNet_NLosc_STR(hidden=50)
    path = ROOT_DIR + '/models/NLosc/STR/x_0={}-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_e, px_e, t_0, t_final, lam)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    end_model.load_state_dict(checkpoint['model_state_dict'])

    plot_analytical_NLosc(init_model, end_model, initial_conditions, end_conditions, t_0, t_final, size)

