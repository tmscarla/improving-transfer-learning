from training import train
from constants import ROOT_DIR
from torch.utils.tensorboard import SummaryWriter
from models import SIRNetwork
import torch

s_0 = 6
i_0 = 3
r_0 = 0
t_final = 40
initial_conditions = [0, [s_0, i_0, r_0]]
beta = round(0.5, 2)
gamma = round(0.5, 2)
train_size = 1000
sir = SIRNetwork(layers=2, hidden=50)

# TRAIN
optimizer = torch.optim.Adam(sir.parameters(), lr=8e-4)
writer = SummaryWriter(
    'runs/' + 's_0={}-i_0={}-r_0={}-t_0={}-t_f={:.2f}_beta={}_gamma={}.pt'.format(s_0,
                                                                                  i_0, r_0,
                                                                                  initial_conditions[0],
                                                                                  t_final, beta, gamma))
# sir, train_losses, run_time, optimizer = train(sir, initial_conditions, t_final=t_final, epochs=int(500),
#                                                 num_batches=10,
#                                                 start_size=start_size, optimizer=optimizer,
#                                                 writer=writer, beta=beta, gamma=gamma)
# torch.save({'model_state_dict': sir.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict()},
#            ROOT_DIR + '/models/SIR/s_0={}-i_0={}-r_0={}'
#                       '-t_0={}-t_f={:.2f}_beta={}_gamma={}.pt'.format(s_0,
#                                                                       i_0, r_0,
#                                                                       initial_conditions[
#                                                                           0],
#                                                                       t_final, beta,
#                                                                       gamma))

# LOAD
checkpoint = torch.load(ROOT_DIR + '/models/SIR/s_0={}-i_0={}-r_0={}'
                                   '-t_0={}-t_f={:.2f}_beta={}_gamma={}.pt'.format(s_0,
                                                                                   i_0,
                                                                                   r_0,
                                                                                   initial_conditions[
                                                                                       0],
                                                                                   t_final,
                                                                                   beta,
                                                                                   gamma))

sir.load_state_dict(checkpoint['model_state_dict'])

perturbation = 0.1
s_1 = s_0 + perturbation * s_0
i_1 = i_0 + perturbation * i_0
r_1 = r_0 + perturbation * r_0
initial_conditions = [0, [s_1, i_1, r_1]]
optimizer = torch.optim.Adam(sir.parameters(), lr=8e-4)
selection = 'inv_exponential'

writer = SummaryWriter(
    'runs/' + 's_0={}-i_0={}-r_0={}-t_0={}-t_f={:.2f}_beta={}_gamma={}_sel={}.pt'.format(s_1,
                                                                                         i_1, r_1,
                                                                                         initial_conditions[0],
                                                                                         t_final, beta, gamma,
                                                                                         selection))

sir, _, _, _ = train(sir, initial_conditions, t_final=t_final, epochs=int(500),
                      num_batches=10,
                      train_size=train_size, optimizer=optimizer, selection=selection,
                      perc=0.5, val_size=train_size,
                      writer=writer, beta=beta, gamma=gamma)

checkpoint = torch.load(ROOT_DIR + '/models/SIR/s_0={}-i_0={}-r_0={}'
                                   '-t_0={}-t_f={:.2f}_beta={}_gamma={}.pt'.format(s_0,
                                                                                   i_0,
                                                                                   r_0,
                                                                                   initial_conditions[
                                                                                       0],
                                                                                   t_final,
                                                                                   beta,
                                                                                   gamma))
sir.load_state_dict(checkpoint['model_state_dict'])
optimizer = torch.optim.Adam(sir.parameters(), lr=8e-4)

writer = SummaryWriter(
    'runs/' + 's_0={}-i_0={}-r_0={}-t_0={}-t_f={:.2f}_beta={}_gamma={}_random.pt'.format(s_1,
                                                                                         i_1, r_1,
                                                                                         initial_conditions[0],
                                                                                         t_final, beta, gamma))

# sir, _, _, optimizer = train(sir, initial_conditions, t_final=t_final, epochs=int(500),
#                              num_batches=10,
#                              start_size=int(start_size * 0.5), val_size=start_size, optimizer=optimizer, writer=writer,
#                              beta=beta, gamma=gamma)
