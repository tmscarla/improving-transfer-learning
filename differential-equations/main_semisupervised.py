from training import train_semisupervised
from constants import ROOT_DIR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
from models import SIR_solution, SIRNetwork

if __name__ == '__main__':

    ### Initial Conditions ###
    N = 1
    rescaling_factor = 1

    infected = 0.01
    susceptible = N - infected
    recovered = 0

    s_0 = susceptible / N * rescaling_factor
    i_0 = infected / N * rescaling_factor
    r_0 = 0

    point_1 = 2.5; known_1 = [0.9, 0.1, 0.0]
    sum_1 = sum(known_1)
    point_2 = 5; known_2 = [0.5, 0.5, 0.0]
    sum_2 = sum(known_2)
    point_3 =7.5; known_3 = [0.1, 0.9, 0.1]
    sum_3 = sum(known_3)

    known_points = {
        point_1: [k / sum_1 for k in known_1],
        point_2: [k / sum_2 for k in known_2],
        point_3: [k / sum_2 for k in known_3]
    }

    assert i_0 + s_0 + r_0 == rescaling_factor

    t_final = 20
    initial_conditions = [0, [s_0, i_0, r_0]]
    beta = round(1, 2)
    gamma = round(0.01, 2)
    train_size = 1800

    ### Scipy solver solution ###
    t = np.linspace(0, t_final, t_final)
    s_p, i_p, r_p = SIR_solution(t, s_0, i_0, r_0, beta, gamma)

    ### Init model ###
    sir = SIRNetwork(layers=4, hidden=50)

    try:
        # It tries to load the model, otherwise it trains it
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_ss/s_0={:.2f}-i_0={:.2f}-r_0={:.2f}'
                       '-t_0={}-t_f={:.2f}_beta={}_gamma={}_known_1={}.pt'.format(s_0,
                                                                                  i_0, r_0,
                                                                                  initial_conditions[0],
                                                                                  t_final, beta,
                                                                                  gamma, known_1))
    except FileNotFoundError:
        # TRAIN
        optimizer = torch.optim.Adam(sir.parameters(), lr=8e-4)
        writer = SummaryWriter(
            'runs/' + 'ss_s_0={:.2f}-i_0={:.2f}-r_0={:.2f}-'
                      't_0={:.2f}-t_f={:.2f}_beta={}_gamma={}_known_1={}.pt'.format(s_0,
                                                                                    i_0, r_0,
                                                                                    initial_conditions[
                                                                                        0],
                                                                                    t_final, beta,
                                                                                    gamma, known_1))
        sir, train_losses, run_time, optimizer = train_semisupervised(sir, initial_conditions, t_final=t_final,
                                                                      epochs=int(500),
                                                                      num_batches=10, known_points=known_points,
                                                                      train_size=train_size, optimizer=optimizer,
                                                                      val_size=10 * train_size,
                                                                      writer=writer, beta=beta, gamma=gamma)
        torch.save({'model_state_dict': sir.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   ROOT_DIR + '/models/SIR_ss/s_0={:.2f}-i_0={:.2f}-r_0={:.2f}'
                              '-t_0={}-t_f={:.2f}_beta={}_gamma={}_known_1={}.pt'.format(s_0,
                                                                                         i_0, r_0,
                                                                                         initial_conditions[0],
                                                                                         t_final,
                                                                                         beta,
                                                                                         gamma, known_1))
        # LOAD
        checkpoint = torch.load(
            ROOT_DIR + '/models/SIR_ss/s_0={:.2f}-i_0={:.2f}-r_0={:.2f}'
                       '-t_0={}-t_f={:.2f}_beta={}_gamma={}_known_1={}.pt'.format(s_0,
                                                                                  i_0, r_0,
                                                                                  initial_conditions[0],
                                                                                  t_final, beta,
                                                                                  gamma, known_1))

    sir.load_state_dict(checkpoint['model_state_dict'])

    ### Test between 0 and t_final ###
    grid = torch.arange(0, t_final, out=torch.FloatTensor()).reshape(-1, 1)
    t_dl = DataLoader(dataset=grid, batch_size=1, shuffle=False)
    s_hat = []
    i_hat = []
    r_hat = []

    for i, t in enumerate(t_dl, 0):
        # Network solutions
        s, i, r = sir.parametric_solution(t, initial_conditions)
        s_hat.append(s.item())
        i_hat.append(i.item())
        r_hat.append(r.item())

    ### Plot network solutions ###
    plt.figure(figsize=(15, 7))
    plt.plot(range(len(s_hat)), s_hat, label='Susceptible')
    plt.plot(range(len(i_hat)), i_hat, label='Infected')
    plt.plot(range(len(r_hat)), r_hat, label='Recovered')
    plt.plot(range(len(s_p)), s_p, label='Susceptible - Scipy', linestyle='--')
    plt.plot(range(len(i_p)), i_p, label='Infected - Scipy', linestyle='--')
    plt.plot(range(len(r_p)), r_p, label='Recovered - Scipy', linestyle='--')
    plt.title('Solving SIR semisupervised model with Beta = {} | Gamma = {} \n'
              'Starting conditions: S0 = {:.2f} | I0 = {:.2f} | R0 = {:.2f} \n'
              'Known points: \n'
              '{} : {}, {} : {}, {} : {}'.format(beta, gamma, s_0, i_0, r_0,
                                                point_1, known_1, point_2, known_2,
                                                point_3, known_3))
    plt.legend(loc='lower right')
    plt.savefig(
        ROOT_DIR + '/plots/ss_SIR_s0={:.2f}_i0={:.2f}_r0={:.2f}_beta={}_gamma={}_known_1={}.png'.format(s_0, i_0, r_0,
                                                                                                        beta,
                                                                                                        gamma, known_1))
    plt.show()
