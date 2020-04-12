import numpy as np
from torch.utils.tensorboard import SummaryWriter

from training_NNLosc import train_NNLosc
from constants import ROOT_DIR, device
from models import odeNet_NLosc_MM, odeNet_NLosc_modular
import torch
import matplotlib.pyplot as plt
import copy


def explore(start_model, initial_conditions, t_final, end_conditions, start_size, end_size=None,
            steps=1, start_tresh=1e-3, end_tresh=None, tresh_mode='log', writer=None, last=False):
    # Unpack values
    t_0, x_0, px_0, lam_0 = initial_conditions
    t_e, x_e, px_e, lam_e = end_conditions

    # Compute total distance and perturbation for each step in both directions
    distance = np.square((x_e - x_0) ** 2 + (px_e - px_0) ** 2)
    pert_x = ((x_e - x_0) / steps)
    pert_px = ((px_e - px_0) / steps)

    if end_size is not None:
        if tresh_mode == 'lin':
            sizes = np.linspace(start_size, end_size, steps)
        elif tresh_mode == 'log':
            sizes = np.logspace(np.log10(start_size), np.log10(end_size))
    else:
        sizes = [start_size] * steps

    # Set the range of tresholds and sizes according to the start and end value and the approach
    if end_tresh is not None:
        if tresh_mode == 'lin':
            tresholds = np.linspace(start_tresh, end_tresh, steps)
        elif tresh_mode == 'log':
            tresholds = np.logspace(np.log10(start_tresh), np.log10(end_tresh))
    else:
        tresholds = [start_tresh] * steps

    # Train the final point (x_e, px_e) for the entire number of epochs
    if last:
        tresholds[-1] = float('-inf')

    # Actual train for each step
    total_losses, start_epoch = [], 0
    for s in range(1, steps + 1):
        print('##### STEP: {} #####'.format(s))
        conditions = [t_0, x_0 + (s * pert_x), px_0 + (s * pert_px), lam_0]
        optimizer = torch.optim.Adam(start_model.parameters(), lr=8e-4)
        start_model, train_losses, _, _ = train_NNLosc(conditions, t_final, lam_0, hidden=50, epochs=int(5e4),
                                                       train_size=sizes[s - 1], optimizer=optimizer,
                                                       start_model=start_model, start_epoch=start_epoch,
                                                       treshold=tresholds[s - 1], writer=writer)
        total_losses += train_losses
        start_epoch += len(train_losses)

    return total_losses, distance


def explore_architecture(start_model, initial_conditions, t_final, end_conditions, treshold, steps, train_size,
                         writer):
    # Unpack values
    t_0, x_0, px_0, lam_0 = initial_conditions
    t_e, x_e, px_e, lam_e = end_conditions

    # Compute total distance and perturbation for each step in both directions
    distance = np.square((x_e - x_0) ** 2 + (px_e - px_0) ** 2)
    pert_x = ((x_e - x_0) / steps)
    pert_px = ((px_e - px_0) / steps)

    # Actual train for each step
    total_losses, start_epoch = [], 0
    for s in range(1, steps + 1):
        print('##### STEP: {} #####'.format(s))
        conditions = [t_0, x_0 + (s * pert_x), px_0 + (s * pert_px), lam_0]
        optimizer = torch.optim.Adam(start_model.parameters(), lr=8e-4)

        # Change last layer
        start_model.ffn._modules[max(start_model.ffn._modules.keys())] = torch.nn.Linear(32, 2)

        start_model, train_losses, _, _ = train_NNLosc(conditions, t_final, lam_0, hidden=None, epochs=int(5e4),
                                                       train_size=train_size, optimizer=optimizer,
                                                       start_model=start_model, start_epoch=start_epoch,
                                                       treshold=treshold, writer=writer)
        total_losses += train_losses
        start_epoch += len(train_losses)

    return total_losses, distance


def run_exp():
    initial_conditions = [0.0, 2.0, 1.0, 1.0]
    end_conditions = [0.0, 1.3, 1.0, 1.0]
    train_size, steps = 200, 20
    start_tresh, end_tresh = 1e-3, 1e-6
    tresh_mode = 'lin'

    model = odeNet_NLosc_MM(hidden=50)
    path = ROOT_DIR + '/models/NLosc/x_0={}-px_0={}-t_0={}-t_f=12.57.pt'.format(initial_conditions[1],
                                                                                initial_conditions[2],
                                                                                initial_conditions[0])

    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    writer = SummaryWriter(
        'runs/' + 'NLoscExp_x0={}_px0={}_xe_{}_pxe={}_size={}'
                  '_steps={}_start_tr={}_end_tr={}_mode={}'.format(initial_conditions[1], initial_conditions[2],
                                                                   end_conditions[1], end_conditions[2], train_size,
                                                                   steps, start_tresh, end_conditions, tresh_mode))

    total_losses, distance = explore(start_model=model, initial_conditions=initial_conditions, t_final=4 * np.pi,
                                     start_size=train_size, end_conditions=end_conditions, steps=steps,
                                     start_tresh=start_tresh, end_tresh=end_tresh, tresh_mode=tresh_mode,
                                     writer=writer, end_size=200)


def run_exp_architecture():
    initial_conditions = [0.0, 1.3, 1.0, 1.0]
    end_conditions = [0.0, 1.5, 2.5, 1.0]
    train_size, steps, treshold = 200, 20, 1e-3

    hidden, layers = 32, 3
    model = odeNet_NLosc_modular(hidden=hidden, layers=layers)
    path = ROOT_DIR + '/models/NLosc/modular/' \
                      'h={}-l={}-x_0={}-px_0={}-t_0={}-t_f=12.57.pt'.format(hidden, layers, initial_conditions[1],
                                                                            initial_conditions[2],
                                                                            initial_conditions[0])

    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_state_dict'])
    writer = SummaryWriter(
        'runs/' + 'NLoscExpArch_x0={}_px0={}_xe_{}_pxe={}_size={}'
                  '_steps={}'.format(initial_conditions[1], initial_conditions[2],
                                     end_conditions[1], end_conditions[2], train_size,
                                     steps))

    total_losses, distance = explore_architecture(start_model=model, initial_conditions=initial_conditions,
                                                  t_final=4 * np.pi, end_conditions=end_conditions,
                                                  treshold=treshold, steps=steps, train_size=train_size,
                                                  writer=writer)


if __name__ == '__main__':
    run_exp_architecture()


    # x0, px0, xe, pxe = initial_conditions[1], initial_conditions[2], end_conditions[1], end_conditions[2]
    # plt.figure(figsize=(12, 8))
    # plt.title('NLosc fine-tuning vs scratch\n'
    #           'x0 = {} | px0 = {}\n'
    #           'xe = {} | pxe = {}'.format(x0, px0, xe, pxe))
    # plt.plot(range(len(total_losses)), total_losses, 'b', linewidth=2,
    #          linestyle='-')
    # plt.ylabel('loss')
    # plt.xlabel('epochs')
    # plt.legend()
    #
    # plt.savefig(
    #     ROOT_DIR + '/plots/exploration_x0={}_px0={}_xe={}'
    #                '_pxe={}_size={}_end_size={}_steps={}_start_tr={}'
    #                '_end_tr={}_mode={}.png'.format(x0, px0, xe, pxe, train_size, 200, steps, start_tresh, end_tresh,
    #                                                tresh_mode))
    # plt.show()
