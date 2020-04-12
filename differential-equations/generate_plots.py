import numpy as np
from constants import ROOT_DIR
import torch

from losses import hamiltonian_eq_loss
from models import odeNet_NLosc_MM
import matplotlib.pyplot as plt
import matplotlib
from constants import device
from tqdm import tqdm
import pandas as pd

matplotlib.rcParams['figure.dpi'] = 300


def plot_loss_from_csv(path):
    dataframe = pd.read_csv(path)
    steps, values = dataframe['Step'].to_list(), dataframe['Value'].to_list()
    values = list(map(np.log, values))
    plt.figure(figsize=(12, 12))
    plt.title('Loss trend - model trained on fixed conditions\n'
              'x(0) = 1.0 | p(0) = 1.0', fontsize=20)
    plt.plot(steps, values, linewidth=2, linestyle='-', color='#cc0000')
    plt.ylabel('LogLoss', fontsize=20)
    plt.xlabel('Epochs', fontsize=20)
    plt.legend()
    plt.savefig(ROOT_DIR + '/plots/nonlinear_oscillator_loss.png')
    plt.show()


def plot_single_NLosc(model, initial_conditions, t_0, t_final, size):
    plt.figure(figsize=(12, 12))
    plt.title('Phase space linear oscillator\n'
              'x(0) = 1.0 | p(0) = 1.0', fontsize=20)
    t = torch.linspace(t_0, t_final, size).reshape(-1, 1)
    x, px = model.parametric_solution(t, initial_conditions)
    x, px = x.detach().numpy(), px.detach().numpy()
    plt.plot(x, px, linewidth=2, linestyle='-', color='#3366ff')
    plt.ylabel('p', fontsize=24)
    plt.xlabel('x', fontsize=24)
    plt.legend()
    plt.savefig(ROOT_DIR + '/plots/phase_space_{}.png'.format(initial_conditions))
    plt.show()


def plot_multiple_NLosc(models, names, conditions, t_0, t_final, size, save_name=''):
    plt.figure(figsize=(12, 12))
    plt.title('Phase space NLosc')
    colors = ['b', 'g', 'r', 'm', 'k', 'y', 'c']
    for i in range(len(models)):
        t = torch.linspace(t_0, t_final, size).reshape(-1, 1)
        x, px = models[i].parametric_solution(t, conditions[i])
        x, px = x.detach().numpy(), px.detach().numpy()
        if i == 1:
            # z, px = models[i-1].parametric_solution(t, conditions[i-1])
            # z, px = z.detach().numpy(), px.detach().numpy()
            # z, px = z*np.sqrt(3/2), px*np.sqrt(3/2)
            linestyle = '-'
        else:
            linestyle = 'dotted'
        plt.plot(x, px, colors[i], linewidth=2, label=names[i], alpha=0.7, linestyle=linestyle)
    plt.ylabel('px')
    plt.xlabel('z')
    plt.legend()
    plt.savefig(ROOT_DIR + '/plots/phase_space_{}.png'.format(save_name))
    plt.show()


def smooth(scalars, weight):
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value

    return smoothed


def plot_solutions(source, target, x_0, px_0, x_e, px_e, t_0, t_final, lam, size):
    models = [source, target]
    names = ['pre-trained', 'from scratch']
    conditions = [[t_0, x_e, px_e, t_final], [t_0, x_e, px_e, t_final]]
    colors = ['#3366ff', '#cc0000']
    t = torch.linspace(t_0, t_final, size).reshape(-1, 1)
    t.requires_grad = True

    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_xlabel('t', fontsize=18)
    ax1.set_ylabel('x', fontsize=18)

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_xlabel('t', fontsize=18)
    ax2.set_ylabel('p', fontsize=18)

    for i in range(len(models)):
        x, px = models[i].parametric_solution(t, conditions[i])
        if i == 0:
            loss = hamiltonian_eq_loss(t, x, px, lam)
        x, px = x.detach().numpy(), px.detach().numpy()
        t_n = t.detach().numpy()
        ax1.plot(t_n, x, linewidth=2, color=colors[i], label=names[i], alpha=0.7, linestyle='--')
        ax2.plot(t_n, px, linewidth=2, color=colors[i], label=names[i], alpha=0.7, linestyle='--')

    ax1.legend()
    ax2.legend()

    st = fig.suptitle('Solving Nonlinear oscillator model\n'
                      'Starting conditions: x(0) = 2.0 | p(0) = 2.0\n'
                      'Loss: {}'.format(loss))

    fig.tight_layout()
    st.set_y(0.95)
    fig.subplots_adjust(top=0.82)

    plt.savefig('img.png')
    fig.show()


if __name__ == '__main__':

    file_name = 'NLosc_init=[1.5, 1.7]_bundle_finetuning.csv'
    df = pd.read_csv(ROOT_DIR + '/plots/csv/{}'.format(file_name))
    steps_fine_1, values_fine_1 = df['Step'].to_numpy(), df['Value'].to_numpy()
    values_fine_1 = [np.log(v) for v in values_fine_1]
    values_fine_1 = smooth(values_fine_1, 0.9)

    file_name = 'NLosc_init=[1.5, 1.7]_bundle.csv'
    df = pd.read_csv(ROOT_DIR + '/plots/csv/{}'.format(file_name))
    steps_scratch_1, values_scratch_1 = df['Step'].to_numpy(), df['Value'].to_numpy()
    values_scratch_1 = [np.log(v) for v in values_scratch_1]
    values_scratch_1 = smooth(values_scratch_1, 0.9)

    file_name = 'NLosc_init=[2.2, 2.5]_bundle_finetuning.csv'
    df = pd.read_csv(ROOT_DIR + '/plots/csv/{}'.format(file_name))
    steps_fine_2, values_fine_2 = df['Step'].to_numpy(), df['Value'].to_numpy()
    values_fine_2 = [np.log(v) for v in values_fine_2]
    values_fine_2 = smooth(values_fine_2, 0.9)

    file_name = 'NLosc_init=[2.2, 2.5]_bundle.csv'
    df = pd.read_csv(ROOT_DIR + '/plots/csv/{}'.format(file_name))
    steps_scratch_2, values_scratch_2 = df['Step'].to_numpy(), df['Value'].to_numpy()
    values_scratch_2 = [np.log(v) for v in values_scratch_2]
    values_scratch_2 = smooth(values_scratch_2, 0.9)

    fig = plt.figure(figsize=(15, 5))
    # st = fig.suptitle('Accuracy trend of a baseline pre-trained on CIFAR 10 dataset\n'
    #                   'and finetuned on a distorted version with embedding shift = 0.0\n'
    #                   'Samples selected according to error-driven criterion')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('Loss trend - model trained on bundle of initial conditions\n'
                  'x(0) bundle = [1.5, 1.7] | p(0) bundle = [1.5, 1.7]')
    ax1.plot(steps_scratch_1, values_scratch_1, linewidth=1, color='#cc0000', label='from scratch')
    ax1.plot(steps_fine_1, values_fine_1, linewidth=1, color='#3366ff', label='finetuned')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('LogLoss', fontsize=12)
    ax1.legend()
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Loss trend - model trained on bundle of initial conditions\n'
                  'x(0) bundle = [2.2, 2.5] | p(0) bundle = [2.0, 2.2]')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('LogLoss', fontsize=12)
    ax2.plot(steps_scratch_2, values_scratch_2, linewidth=1, color='#cc0000', label='from scratch')
    ax2.plot(steps_fine_2, values_fine_2, linewidth=1, color='#3366ff', label='finetuned')
    ax2.legend()

    # fig.tight_layout()

    plt.savefig('img.png')
    fig.show()
