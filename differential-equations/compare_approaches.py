import torch
import numpy as np
from models import odeNet_NLosc_MM
from constants import ROOT_DIR
from training_NNLosc import train_NNLosc
from exploration import explore
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
from fine_tuning import find_tau


def compare_solutions_NLosc(models, names, conditions, t_0, t_final, size, save_name):
    # Generate points and figure
    t = torch.linspace(t_0, t_final, size).reshape(-1, 1)
    colors = ['b', 'g', 'r', 'm', 'k', 'y', 'c']
    plt.figure(figsize=(12, 8))
    plt.title('NLosc | z solutions')

    # Generate plot for each source_model
    for i in range(len(models)):
        x, px = models[i].parametric_solution(t, conditions[i])
        t_n = t.detach().numpy()
        x, px = x.detach().numpy(), px.detach().numpy()
        plt.plot(t_n, x, colors[i], linewidth=2, linestyle='--', label=names[i], alpha=0.75)

    plt.ylabel('z')
    plt.xlabel('t')
    plt.legend()
    plt.savefig(ROOT_DIR + '/plots/solutions_{}.png'.format(save_name))
    plt.show()


if __name__ == '__main__':
    x_0, px_0, lam = 1.0, 1.0, 1
    x_e, px_e = 1.5, 1.5
    t_0, t_final, train_size = 0., 4 * np.pi, 200
    epochs = int(5e4)
    models, names, conditions = [], [], []

    # TODO CHANGE TAU
    # start_model = odeNet_NLosc_MM(hidden=50, tau=0)
    # optimizer = torch.optim.Adam(start_model.parameters(), lr=8e-4)

    # EXPLORATION
    # print('Finetuning with exploration...')
    # start_model = odeNet_NLosc_MM(hidden=50)
    # optimizer = torch.optim.Adam(start_model.parameters(), lr=8e-4)
    # path = ROOT_DIR + '/models/NLosc/x_0={}-px_0={}-t_0={}-t_f={:.2f}.pt'.format(x_0, px_0, t_0, t_final)
    # checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    # start_model.load_state_dict(checkpoint['model_state_dict'])
    #
    # epochs, steps = int(5e4), 20
    # start_tr, end_tr, mode = 1e-3, 1e-6, 'lin'
    # writer = SummaryWriter('runs/' + 'NLosc_x0={}_px0={}_xe={}_pxe={}_steps={}'
    #                                  '_start_tr={}_end_tr={}_mode={}'.format(x_0, px_0, x_e, px_e, steps,
    #                                                                          start_tr, end_tr, mode))
    #
    # initial_conditions, end_conditions = [t_0, x_0, px_0, lam], [t_0, x_e, px_e, lam]
    # explore(start_model, initial_conditions, t_final, end_conditions, start_size=train_size, steps=steps,
    #         start_tresh=start_tr, end_tresh=end_tr, tresh_mode=mode, writer=writer)

    # TRANSLATIONAL FINETUNING
    # print('\nTranslational finetuning on ({},{})...'.format(x_e, px_e))
    # path = ROOT_DIR + '/models/NLosc/x_0={}-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_0, px_0, t_0, t_final, lam)
    # checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    # trans_model = odeNet_NLosc_MM(hidden=50)
    # trans_model.load_state_dict(checkpoint['model_state_dict'])
    # tau = find_tau(trans_model, initial_conditions=[t_0, x_0, px_0, lam], end_conditions=[t_0, x_e, px_e, lam],
    #                t_final=t_final, epochs=300, lr=1e-2)
    # trans_model.tau = tau
    # optimizer = torch.optim.Adam(trans_model.parameters(), lr=8e-4)
    # initial_conditions = [t_0, x_e, px_e, lam]
    # _, trans_losses, _, _ = train_NNLosc(initial_conditions=initial_conditions, t_final=t_final, lam=lam,
    #                                      hidden=50, epochs=epochs,
    #                                      train_size=train_size, optimizer=optimizer, num_batches=1,
    #                                      start_model=trans_model, val_size=train_size,
    #                                      selection=None, perc=1., additional_comment='_translational_N')
    # models.append(trans_model)
    # names.append('translational_N')
    # conditions.append(initial_conditions)

    # FINETUNING
    print('\nFinetuning from the beginning...')
    path = ROOT_DIR + '/models/NLosc/x_0={}-px_0={}-t_0={}-t_f={:.2f}_lam={}.pt'.format(x_0, px_0, t_0, t_final, lam)
    checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    fine_model = odeNet_NLosc_MM(hidden=50)
    fine_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(fine_model.parameters(), lr=8e-4)

    initial_conditions = [t_0, x_e, px_e, lam]
    _, fine_losses, _, _ = train_NNLosc(initial_conditions=initial_conditions, t_final=t_final, lam=lam,
                                        hidden=50, epochs=epochs,
                                        train_size=train_size, optimizer=optimizer, num_batches=1,
                                        start_model=fine_model, val_size=train_size,
                                        selection=None, perc=1., additional_comment='_finetuning')
    models.append(fine_model)
    names.append('finetuning')
    conditions.append(initial_conditions)

    # FROM SCRATCH
    print('\nFrom scratch...')
    scratch_model = odeNet_NLosc_MM(hidden=50)
    optimizer = torch.optim.Adam(scratch_model.parameters(), lr=8e-4)
    _, scratch_losses, _, _ = train_NNLosc(initial_conditions=initial_conditions, t_final=t_final, lam=lam, hidden=50,
                                           epochs=epochs,
                                           train_size=train_size, optimizer=optimizer, num_batches=1,
                                           start_model=scratch_model, val_size=train_size,
                                           selection=None, perc=1., additional_comment='_scratch')
    models.append(scratch_model)
    names.append('scratch')
    conditions.append(initial_conditions)

    # COMPARE SOLUTIONS
    # compare_solutions_NLosc(models, names, conditions, t_0, t_final, train_size, 'prova')
