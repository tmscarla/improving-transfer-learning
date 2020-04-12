#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:13:26 2020
@author: marios mattheakis

In this code a Hamiltonian Neural Network is designed and employed
to solve a system of two differential equations obtained by Hamilton's
equations for the the Hamiltonian of nonlinear oscillator.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from losses import dfx
from energy_computations import energy, NLosc_exact
from training import train_NNLosc
from alternative_solvers import NLosc_solution, symEuler
from constants import ROOT_DIR
from models import odeNet_NLosc_MM

dtype = torch.float

# Set the initial conditions. The parameter lam controls the nonlinearity
x_0, px_0, lam = 1.3, 1., 1
t_0, t_final, train_size = 0., 4 * np.pi, 200
dt = t_final / train_size
initial_conditions = [t_0, x_0, px_0, lam]
t_num = np.linspace(t_0, t_final, train_size)

# Scipy solver results
E_0, E_ex = NLosc_exact(train_size, x_0, px_0, lam)

# Solution obtained by Scipy solver
x_num, px_num = NLosc_solution(train_size, t_num, x_0, px_0, lam)
E_num = energy(x_num, px_num, lam)

# Train the network
# Here, we use one mini-batch. No significant different in using more
train_size, neurons, epochs, lr, num_batches = 200, 50, int(5e1), 8e-3, 1
# model_0 = odeNet_NLosc_MM(hidden=50)
# betas = (0.999, 0.9999)
# optimizer = torch.optim.Adam(model_0.parameters(), lr=lr, betas=betas)
# model_0, loss_0, runTime_0 = train(initial_conditions=initial_conditions, t_final=t_final, lam=lam, hidden=neurons,
#                                    epochs=epochs, optimizer=optimizer,
#                                    start_size=start_size, num_batches=num_batches,
#                                    start_model=model_0)
#

perturbations = np.linspace(0.01, 0.1, 3)
# quarters = [[0, t_final/4], [t_final/4, t_final/2], [t_final/2, t_final*3/4], [t_final * 3/4, t_final]]

# Train with just one quarter at a time

# for q in quarters:
for train_size in [50, 100, 200]:
    for p in perturbations:
        start_model = odeNet_NLosc_MM(hidden=50)
        optimizer = torch.optim.Adam(start_model.parameters(), lr=8e-4, amsgrad=True)
        path = ROOT_DIR + '/models/NLosc/x_0={}-px_0={}-t_0={}-t_f={:.2f}.pt'.format(x_0, px_0, t_0, t_final)

        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)

        start_model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        epochs = int(1e4)
        initial_conditions = [t_0, x_0 + p, px_0 + p, lam]
        train_NNLosc(initial_conditions=initial_conditions, t_final=t_final, lam=lam, hidden=neurons, epochs=epochs,
                     train_size=train_size, optimizer=optimizer, num_batches=num_batches, start_model=start_model,
                     val_size=200, selection=None, perc=0.25)

# Test
test_size = train_size
t_test = torch.linspace(t_0, t_final, test_size)
t_test = t_test.reshape(-1, 1)
t_test.requires_grad = True
t_net = t_test.detach().numpy()

x, px = start_model.parametric_solution(t_test, initial_conditions)

# Here we calculate the maximum loss in time
x_d, px_d = dfx(t_test, x), dfx(t_test, px)  # derivatives obtained by back-propagation
fx = x_d - px
fpx = px_d + x + x.pow(3)
ell_sq = fx.pow(2) + fpx.pow(2)
ell_max = np.max(np.sqrt(ell_sq.data.numpy()))
print('The maximum in time loss is ', ell_max)

train_size_sym = train_size - 1
E_s, x_s, p_s, t_s = symEuler(train_size_sym, x_0, px_0, t_0, t_final, lam)
train_size_sym_100 = 100 * train_size
E_s100, x_s100, p_s100, t_s100 = symEuler(train_size_sym_100, x_0, px_0, t_0, t_final, lam)
