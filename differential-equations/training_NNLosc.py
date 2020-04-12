import torch
import time
import copy
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from models import odeNet_NLosc_MM
from torch.utils.data import DataLoader
from losses import *
from constants import device
from utils import generate_dataloader, generate_grid


def train_NNLosc(initial_conditions, t_final, lam, hidden, epochs, train_size, optimizer,
                 num_batches=1, start_model=None, t_0_train=None, t_final_train=None, val_size=None,
                 selection=None, perc=1.0, treshold=float('-inf'), additional_comment='', verbose=True,
                 writer=None, start_epoch=0, grid=None, perturb=True, Mblock=False, explorer=None, grid_explorer=None):
    if not start_model:
        model = odeNet_NLosc_MM(hidden)
    else:
        model = start_model

    if selection is None:
        perc = 1.0

    model = model.to(device)
    model.train()
    best_model = model

    # Create writer if none is provided
    if writer is None:
        writer = get_writer(t_0_train, t_final_train, initial_conditions, int(train_size * perc),
                            selection, additional_comment)

    train_losses = []
    val_losses = []
    min_loss = 1

    t_0 = initial_conditions[0]

    # Grid of points to use for training
    if t_0_train is None or t_final_train is None:
        t_0_train, t_final_train = t_0, t_final

    # Points selection
    if grid is None:
        grid = generate_grid(selection, model, initial_conditions, t_0_train, t_final_train, train_size, perc=perc)

    if Mblock:
        grid = torch.cat((grid, grid_explorer))

    start_time = time.time()

    for epoch in tqdm(range(epochs), desc='Training Hamiltonian NN on NLosc', disable=not verbose):

        # Generate DataLoader
        batch_size = int(train_size / num_batches)
        t_dataloader = generate_dataloader(grid, t_0_train, t_final_train, batch_size, perturb=perturb)

        # Perturbing the evaluation points & forcing t[0]=t0
        train_epoch_loss = 0.0

        for i, data in enumerate(t_dataloader, 0):
            #  Network solutions
            data = data.to(device)
            x, px = model.parametric_solution(data, initial_conditions)

            # Loss function defined by Hamilton Eqs. (symplectic): Writing explicitely the Eqs (faster)
            batch_loss = hamiltonian_eq_loss(data, x, px, lam)

            # Optimization
            batch_loss.backward(retain_graph=True)  # True

            optimizer.step()

            train_epoch_loss += batch_loss.item()
            optimizer.zero_grad()

        if not val_size:
            val_size = train_size

        if epoch % 500 == 0:
            val_epoch_loss = validate_NNLosc(model, initial_conditions, lam, t_0, t_final, val_size, num_batches)

        # Keep the loss function history
        train_losses.append(train_epoch_loss)
        val_losses.append(val_epoch_loss)

        # Update writer
        writer.add_scalar('Loss/train', train_epoch_loss, start_epoch + epoch)
        writer.add_scalar('Loss/val', val_epoch_loss, start_epoch + epoch)

        # Keep the best source_model (lowest loss) by using a deep copy
        if epoch > 0.8 * epochs and train_epoch_loss < min_loss:
            best_model = copy.deepcopy(model)
            min_loss = train_epoch_loss

        if train_epoch_loss < treshold:
            break

    final_time = time.time()
    run_time = final_time - start_time
    return best_model, train_losses, run_time, optimizer


def validate_NNLosc(model, initial_conditions, lam, t_0, t_final, val_size, num_batches):
    # Grid of points to use for validation
    t = torch.linspace(t_0, t_final, val_size).reshape(-1, 1)
    batch_size = int(val_size / num_batches)
    t.requires_grad = True
    t_dataloader = DataLoader(dataset=t, batch_size=batch_size, shuffle=True)

    val_loss = 0.0

    for i, data in enumerate(t_dataloader, 0):
        data = data.to(device)

        # Network solutions
        x, px = model.parametric_solution(data, initial_conditions)

        # Loss function defined by Hamilton Eqs. (symplectic): Writing explicitely the Eqs (faster)
        val_epoch_loss = hamiltonian_eq_loss(data, x, px, lam)

        # Loss function defined by Hamilton Eqs. (symplectic): Calculating with auto-diff the Eqs (slower)
        # Ltot = hamEqs_Loss_byH(t_mb,z,y,px,py,lam)

        val_loss += val_epoch_loss.item()
    return val_loss


def get_writer(t_0_train, t_final_train, initial_conditions, train_size, selection, additional_comment):
    init = [round(initial_conditions[1], 2), round(initial_conditions[2], 2)]

    if selection is None:
        if not t_0_train and not t_final_train:
            writer = SummaryWriter(
                'runs/' + 'NLosc_init={}_train-size={}{}'.format(init, train_size, additional_comment))
        else:
            writer = SummaryWriter(
                'runs/' + 'NLosc_init={}_t0={:.2f}_t-f_{:.2f}_train_size={}{}'.format(init, t_0_train,
                                                                                      t_final_train,
                                                                                      train_size, additional_comment))
    else:
        if not t_0_train and not t_final_train:
            writer = SummaryWriter(
                'runs/' + 'NLosc_init={}_train-size={}_{}{}'.format(init, train_size, selection, additional_comment))
        else:
            writer = SummaryWriter(
                'runs/' + 'NLosc_init={}_t0={:.2f}_t-f_{:.2f}_train_size={}_{}{}'.format(init, t_0_train,
                                                                                         t_final_train,
                                                                                         train_size, selection,
                                                                                         additional_comment
                                                                                         ))
    return writer


def train_NNLosc_points(model, initial_conditions, t_final, lam, hidden, epochs, optimizer,
                        grid, additional_comment='points', verbose=True, writer=None):
    model = model.to(device)
    model.train()
    best_model = model
    t_0 = initial_conditions[0]

    # Create writer if none is provided
    if writer is None:
        writer = get_writer(t_0, t_final, initial_conditions, len(grid),
                            None, additional_comment)

    train_losses = []
    min_loss = 1
    start_time = time.time()

    for epoch in tqdm(range(epochs), desc='Training Hamiltonian NN on NLosc', disable=not verbose):

        # Generate DataLoader
        t_dataloader = generate_dataloader(grid, t_0, t_final, len(grid), perturb=False)

        # Perturbing the evaluation points & forcing t[0]=t0
        train_epoch_loss = 0.0

        for i, data in enumerate(t_dataloader, 0):
            #  Network solutions
            data = data.to(device)
            x, px = model.parametric_solution(data, initial_conditions)

            # Loss function defined by Hamilton Eqs. (symplectic): Writing explicitely the Eqs (faster)
            batch_loss = hamiltonian_eq_loss(data, x, px, lam)

            # Optimization
            batch_loss.backward(retain_graph=True)  # True

            optimizer.step()

            train_epoch_loss += batch_loss.item()
            optimizer.zero_grad()

        # Keep the loss function history
        train_losses.append(train_epoch_loss)

        # Update writer
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)

        # Keep the best source_model (lowest loss) by using a deep copy
        if epoch > 0.8 * epochs and train_epoch_loss < min_loss:
            best_model = copy.deepcopy(model)
            min_loss = train_epoch_loss

    final_time = time.time()
    run_time = final_time - start_time
    return best_model, train_losses, run_time, optimizer


def train_NNLosc_bundle(model, x_l, x_r, p_l, p_r, t_0, t_final, lam, epochs, optimizer, train_size,
                        additional_comment='_bundle', verbose=True, writer=None):
    model = model.to(device)
    model.train()
    best_model = model

    # Create writer if none is provided
    if writer is None:
        writer = get_writer(t_0, t_final, [0, x_l, x_r, 0], train_size,
                            None, additional_comment)

    train_losses = []
    min_loss = 1
    start_time = time.time()

    # Generate grid_explorer for t
    grid = torch.linspace(t_0, t_final, train_size).reshape(-1, 1)
    x_space = torch.linspace(x_l, x_r, train_size)
    p_space = torch.linspace(p_l, p_r, train_size)

    for epoch in tqdm(range(epochs), desc='Training Hamiltonian NN on NLosc bundle', disable=not verbose):

        # Generate DataLoader
        t_dataloader = generate_dataloader(grid, t_0, t_final, len(grid), perturb=False)

        # Perturbing the evaluation points & forcing t[0]=t0
        train_epoch_loss = 0.0

        for i, data in enumerate(t_dataloader, 0):
            #  Network solutions
            data = data.to(device)
            x_i = np.random.choice(x_space, 1)[0]
            px_i = np.random.choice(p_space, 1)[0]

            x, px = model.parametric_solution(data, [t_0, x_i, px_i, lam])

            # Loss function defined by Hamilton Eqs. (symplectic): Writing explicitely the Eqs (faster)
            batch_loss = hamiltonian_eq_loss(data, x, px, lam)

            # Optimization
            batch_loss.backward(retain_graph=True)  # True

            optimizer.step()

            train_epoch_loss += batch_loss.item()
            optimizer.zero_grad()

        # Keep the loss function history
        train_losses.append(train_epoch_loss)

        # Update writer
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)

        # Keep the best source_model (lowest loss) by using a deep copy
        if epoch > 0.8 * epochs and train_epoch_loss < min_loss:
            best_model = copy.deepcopy(model)
            min_loss = train_epoch_loss

    final_time = time.time()
    run_time = final_time - start_time
    return best_model, train_losses, run_time, optimizer



def train_NNLosc_Mblock(model, explorer_model, initial_conditions, end_conditions, t_final, lam, epochs, train_size,
                        optimizer, num_batches=1, additional_comment='_Mblock', verbose=True,
                        writer=None, start_epoch=0, perturb=True, grid_explorer=None, gamma=0.9):

    model = model.to(device)
    model.train()
    best_model = model
    t_0 = initial_conditions[0]

    # Create writer if none is provided
    if writer is None:
        writer = get_writer(t_0, t_final, initial_conditions, train_size,
                            None, additional_comment)

    train_losses = []
    min_loss = 1

    grid = torch.linspace(t_0, t_final, train_size).reshape(-1, 1)
    t_explorer_dataloader = generate_dataloader(grid_explorer, t_0, t_final, len(grid_explorer),
                                                perturb=False, shuffle=False)

    start_time = time.time()

    for epoch in tqdm(range(epochs), desc='Training Hamiltonian NN on NLosc Mblock', disable=not verbose):

        batch_size = int(train_size / num_batches)
        t_dataloader = generate_dataloader(grid, t_0, t_final, batch_size, perturb=perturb)

        # Perturbing the evaluation points & forcing t[0]=t0
        train_epoch_loss = 0.0

        # General grid_explorer
        # for i, data in enumerate(t_dataloader, 0):
        #     #  Network solutions
        #     data = data.to(device)
        #     x, px = model.parametric_solution(data, end_conditions)
        #
        #     # Loss function defined by Hamilton Eqs.
        #     eq_loss = hamiltonian_eq_loss(data, x, px, lam)

        # Explorer grid_explorer
        for i, data in enumerate(t_explorer_dataloader, 0):
            data = data.to(device)
            x, px = model.parametric_solution_M(data, initial_conditions)
            x_true, px_true = explorer_model.parametric_solution(data, end_conditions)

            # Loss function
            exp_loss = mblock_loss(x, px, x_true, px_true)

        # total_loss = gamma * eq_loss + (1 - gamma) * exp_loss
        total_loss = exp_loss

        # Optimization
        total_loss.backward(retain_graph=True)

        optimizer.step()

        train_epoch_loss += total_loss.item()
        optimizer.zero_grad()

        # Keep the loss function history
        train_losses.append(train_epoch_loss)

        # Update writer
        writer.add_scalar('Loss/train', train_epoch_loss, start_epoch + epoch)

        # Keep the best source_model (lowest loss) by using a deep copy
        if epoch > 0.8 * epochs and train_epoch_loss < min_loss:
            best_model = copy.deepcopy(model)
            min_loss = train_epoch_loss

    final_time = time.time()
    run_time = final_time - start_time
    return best_model, train_losses, run_time, optimizer
