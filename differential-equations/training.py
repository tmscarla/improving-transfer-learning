import numpy as np
import time
import copy
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torch.utils.data import DataLoader
from losses import *
from constants import device, ROOT_DIR
from utils import generate_dataloader, generate_grid, generate_brownian


def train(model, initial_conditions, t_final, epochs, train_size, optimizer, beta, gamma, decay=0,
          num_batches=1, val_size=None, hack_trivial=False, t_0=0,
          selection=None, perc=1.0, treshold=float('-inf'), additional_comment='', verbose=True, writer=None):
    # Check if any selection criteria on samples has been passed
    if selection is None:
        perc = 1.0

    # Move to device
    model = model.to(device)

    # Train mode
    model.train()

    # Initialize model to return
    best_model = model

    # Tensorboard writer
    if writer is None:
        writer = get_writer(t_0, t_final, initial_conditions, int(train_size * perc),
                            selection, additional_comment)

    # Initialize losses arrays
    train_losses, val_losses, min_loss = [], [], 1

    # Fetch parameters of the differential equation
    t_0, params_0 = initial_conditions[0], initial_conditions[1:]

    # Points selection
    grid = generate_grid(selection, model, initial_conditions, t_0, t_final, train_size, perc=perc)

    start_time = time.time()

    for epoch in tqdm(range(epochs), desc='Training', disable=not verbose):

        # Check if the decay should be increased or not
        # decay = modularize_decay(model, t_final, initial_conditions, decay)

        # Generate DataLoader
        batch_size = int(train_size / num_batches)
        t_dataloader = generate_dataloader(grid, t_0, t_final, batch_size, perturb=True)

        train_epoch_loss = 0.0

        for i, t in enumerate(t_dataloader, 0):
            #  Network solutions
            t = t.to(device)
            s, i, r = model.parametric_solution(t, initial_conditions)

            # Loss computation
            batch_loss = sir_loss(t, s, i, r, beta=beta, gamma=gamma, decay=decay)

            # Hack to prevent the network from solving the equations trivially
            if hack_trivial:
                batch_trivial_loss = trivial_loss(model, t_final, initial_conditions, method='mse')
                batch_loss = batch_loss + batch_trivial_loss

            # Optimization
            batch_loss.backward()
            optimizer.step()
            train_epoch_loss += batch_loss.item()
            optimizer.zero_grad()

        # If not specified, the validation size is the same as the train size
        if not val_size:
            val_size = train_size

        # Once every few epochs, we run validation
        if epoch % 100 == 0:
            val_epoch_loss = validate(model, initial_conditions, beta, gamma, t_final, val_size, num_batches)

        # Keep the loss function history
        train_losses.append(train_epoch_loss)
        # val_losses.append(val_epoch_loss)
        writer.add_scalar('LogLoss/train', np.log(train_epoch_loss), epoch)
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)

        # Keep the best model (lowest loss) by using a deep copy
        if epoch > 0.8 * epochs and train_epoch_loss < min_loss:
            best_model = copy.deepcopy(model)
            min_loss = train_epoch_loss

        # If a treshold is passed, we stop training when it is reached. Notice default value is -inf
        if train_epoch_loss < treshold:
            break

    final_time = time.time()
    run_time = final_time - start_time
    return best_model, train_losses, run_time, optimizer


def train_bundle_params(model, initial_conditions, t_final, epochs, train_size, optimizer, betas, gammas,
                        decay=0, num_batches=1, hack_trivial=False, t_0=0, selection=None, perc=1.0,
                        treshold=float('-inf'),
                        additional_comment='', verbose=True, writer=None):
    # Check if any selection criteria on samples has been passed
    if selection is None:
        perc = 1.0

    # Move to device
    model = model.to(device)

    # Train mode
    model.train()

    # Initialize model to return
    best_model = model

    # Tensorboard writer
    if writer is None:
        writer = get_writer(t_0, t_final, initial_conditions, int(train_size * perc),
                            selection, additional_comment)

    # Initialize losses arrays
    train_losses, val_losses, min_loss = [], [], 1

    # Fetch parameters of the differential equation
    t_0, params_0 = initial_conditions[0], initial_conditions[1:]

    # Points selection
    grid = generate_grid(selection, model, initial_conditions, t_0, t_final, train_size, perc=perc)

    start_time = time.time()

    # Generate betas and gammas to sample from
    betas = torch.linspace(betas[0], betas[1], steps=train_size).reshape(-1, 1)
    gammas = torch.linspace(gammas[0], gammas[1], steps=train_size).reshape(-1, 1)

    for epoch in tqdm(range(epochs), desc='Training', disable=not verbose):

        # Check if the decay should be increased or not
        # decay = modularize_decay(model, t_final, initial_conditions, decay)

        # Generate DataLoader
        batch_size = int(train_size / num_batches)
        t_dataloader = generate_dataloader(grid, t_0, t_final, batch_size, perturb=True)

        train_epoch_loss = 0.0

        for i, t in enumerate(t_dataloader, 0):
            # Sample randomly beta and gamma
            rnd_beta = np.random.randint(betas.shape[0], size=batch_size)
            rnd_gamma = np.random.randint(gammas.shape[0], size=batch_size)
            beta = betas[rnd_beta]
            gamma = gammas[rnd_gamma]

            #  Network solutions
            t = t.to(device)
            s, i, r = model.parametric_solution(t, initial_conditions, beta, gamma, mode='bundle_params')

            # Loss computation
            batch_loss = sir_loss(t, s, i, r, beta=beta, gamma=gamma, decay=decay)

            # Hack to prevent the network from solving the equations trivially
            if hack_trivial:
                batch_trivial_loss = trivial_loss(model, t_final, initial_conditions, method='mse')
                batch_loss = batch_loss + batch_trivial_loss

            # Optimization
            batch_loss.backward()
            optimizer.step()
            train_epoch_loss += batch_loss.item()
            optimizer.zero_grad()

        # Keep the loss function history
        train_losses.append(train_epoch_loss)
        writer.add_scalar('LogLoss/train', np.log(train_epoch_loss), epoch)
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)

        # Keep the best model (lowest loss) by using a deep copy
        if epoch > 0.8 * epochs and train_epoch_loss < min_loss:
            best_model = copy.deepcopy(model)
            min_loss = train_epoch_loss

        # If a treshold is passed, we stop training when it is reached. Notice default value is -inf
        if train_epoch_loss < treshold:
            break

    final_time = time.time()
    run_time = final_time - start_time
    return best_model, train_losses, run_time, optimizer


def train_bundle_total(model, initial_conditions_set, t_final, epochs, train_size, optimizer, betas, gammas,
                       decay=0, num_batches=1, hack_trivial=False, t_0=0, selection=None, perc=1.0, sigma=0,
                       treshold=float('-inf'), additional_comment='', verbose=True, writer=None):
    # Check if any selection criteria on samples has been passed
    if selection is None:
        perc = 1.0

    # Move to device
    model = model.to(device)

    # Train mode
    model.train()

    # Initialize model to return
    best_model = model

    # Fetch parameters of the differential equation
    t_0, initial_conditions_set = initial_conditions_set[0], initial_conditions_set[1:]

    # Initialize losses arrays
    train_losses, val_losses, min_loss = [], [], 1

    # Points selection
    grid = generate_grid(selection, model, initial_conditions_set, t_0, t_final, train_size, perc=perc)

    start_time = time.time()

    # Generate initial conditions, betas and gammas to sample from
    betas = torch.linspace(betas[0], betas[1], steps=train_size).reshape(-1, 1)
    gammas = torch.linspace(gammas[0], gammas[1], steps=train_size).reshape(-1, 1)
    s_0_set = torch.linspace(initial_conditions_set[0][0],
                             initial_conditions_set[0][1], steps=train_size).reshape(-1, 1)

    # Generate brownian noise to add stochasticity
    if sigma:
        brownian_noise = generate_brownian(t_0, t_final, train_size, sigma=sigma)
    else:
        brownian_noise = None

    for epoch in tqdm(range(epochs), desc='Training', disable=not verbose):

        # Check if the decay should be increased or not
        # decay = modularize_decay(model, t_final, initial_conditions, decay)

        # Generate DataLoader
        batch_size = int(train_size / num_batches)
        t_dataloader = generate_dataloader(grid, t_0, t_final, batch_size, perturb=False)

        train_epoch_loss = 0.0

        for i, t in enumerate(t_dataloader, 0):
            # Sample randomly initial conditions, beta and gamma
            rnd_beta = np.random.randint(betas.shape[0], size=batch_size)
            rnd_gamma = np.random.randint(gammas.shape[0], size=batch_size)
            rnd_init_s_0 = np.random.randint(s_0_set.shape[0], size=batch_size)
            beta = betas[rnd_beta]
            gamma = gammas[rnd_gamma]
            s_0 = s_0_set[rnd_init_s_0]
            i_0 = 1 - s_0  # Set i_0 to be 1-s_0 to enforce that the sum of the initial conditions is 1
            r_0 = 0.0  # We fix recovered people at day zero to zero
            initial_conditions = [s_0, i_0, r_0]

            #  Network solutions
            t = t.to(device)
            s, i, r = model.parametric_solution(t, initial_conditions, beta, gamma, mode='bundle_total')

            # Loss computation
            batch_loss = sir_loss(t, s, i, r, beta=beta, gamma=gamma, decay=decay, noise=brownian_noise, sigma=sigma)

            # Hack to prevent the network from solving the equations trivially
            if hack_trivial:
                batch_trivial_loss = trivial_loss(model, t_final, initial_conditions_set, method='mse')
                batch_loss = batch_loss + batch_trivial_loss

            # Optimization
            batch_loss.backward()
            optimizer.step()
            train_epoch_loss += batch_loss.item()
            optimizer.zero_grad()

        # Keep the loss function history
        train_losses.append(train_epoch_loss)
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)
        writer.add_scalar('Loss/Log-train', np.log(train_epoch_loss), epoch)

        # Keep the best model (lowest loss) by using a deep copy
        if epoch > 0.8 * epochs and train_epoch_loss < min_loss:
            best_model = copy.deepcopy(model)
            min_loss = train_epoch_loss

        # If a treshold is passed, we stop training when it is reached. Notice default value is -inf
        if train_epoch_loss < treshold:
            break

        # Backup save
        if epoch % 1000 == 0:
            # Save the model
            import datetime
            timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       ROOT_DIR + '/models/SIR_bundle_total/backup_{}.pt'.format(timestamp))

    final_time = time.time()
    run_time = final_time - start_time
    return best_model, train_losses, run_time, optimizer


def train_bundle_init(model, initial_conditions_set, t_final, epochs, train_size, optimizer, beta, gamma,
                      decay=0, num_batches=1, hack_trivial=False, t_0=0, selection=None, perc=1.0, sigma=0,
                      treshold=float('-inf'), additional_comment='', verbose=True, writer=None):
    # Check if any selection criteria on samples has been passed
    if selection is None:
        perc = 1.0

    # Move to device
    model = model.to(device)

    # Train mode
    model.train()

    # Initialize model to return
    best_model = model

    # Fetch parameters of the differential equation
    t_0, initial_conditions_set = initial_conditions_set[0], initial_conditions_set[1:]

    # Initialize losses arrays
    train_losses, val_losses, min_loss = [], [], 1

    # Points selection
    grid = generate_grid(selection, model, initial_conditions_set, t_0, t_final, train_size, perc=perc)

    start_time = time.time()

    # Generate initial conditions to sample from
    s_0_set = torch.linspace(initial_conditions_set[0][0],
                             initial_conditions_set[0][1], steps=train_size).reshape(-1, 1)

    # Generate brownian noise to add stochasticity
    if sigma:
        brownian_noise = generate_brownian(t_0, t_final, train_size, sigma=sigma)
    else:
        brownian_noise = None

    for epoch in tqdm(range(epochs), desc='Training', disable=not verbose):

        # Check if the decay should be increased or not
        # decay = modularize_decay(model, t_final, initial_conditions, decay)

        # Generate DataLoader
        batch_size = int(train_size / num_batches)
        t_dataloader = generate_dataloader(grid, t_0, t_final, batch_size, perturb=False)

        train_epoch_loss = 0.0

        for i, t in enumerate(t_dataloader, 0):
            # Sample randomly initial conditions
            rnd_init_s_0 = np.random.randint(s_0_set.shape[0], size=batch_size)
            s_0 = s_0_set[rnd_init_s_0]
            i_0 = 1 - s_0  # Set i_0 to be 1-s_0 to enforce that the sum of the initial conditions is 1
            r_0 = 0.0  # We fix recovered people at day zero to zero
            initial_conditions = [s_0, i_0, r_0]

            #  Network solutions
            t = t.to(device)
            s, i, r = model.parametric_solution(t, initial_conditions, beta, gamma, mode='bundle_init')

            # Loss computation
            batch_loss = sir_loss(t, s, i, r, beta=beta, gamma=gamma, decay=decay, noise=brownian_noise, sigma=sigma)

            # Hack to prevent the network from solving the equations trivially
            if hack_trivial:
                batch_trivial_loss = trivial_loss(model, t_final, initial_conditions_set, method='mse')
                batch_loss = batch_loss + batch_trivial_loss

            # Optimization
            batch_loss.backward()
            optimizer.step()
            train_epoch_loss += batch_loss.item()
            optimizer.zero_grad()

        # Keep the loss function history
        train_losses.append(train_epoch_loss)
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)
        writer.add_scalar('Loss/Log-train', np.log(train_epoch_loss), epoch)

        # Keep the best model (lowest loss) by using a deep copy
        if epoch > 0.8 * epochs and train_epoch_loss < min_loss:
            best_model = copy.deepcopy(model)
            min_loss = train_epoch_loss

        # If a treshold is passed, we stop training when it is reached. Notice default value is -inf
        if train_epoch_loss < treshold:
            break

        # Backup save
        if epoch % 1000 == 0:
            # Save the model
            import datetime
            timestamp = datetime.datetime.now().strftime("%H-%M-%S")
            torch.save({'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()},
                       ROOT_DIR + '/models/SIR_bundle_total/backup_{}.pt'.format(timestamp))

    final_time = time.time()
    run_time = final_time - start_time
    return best_model, train_losses, run_time, optimizer


def train_semisupervised(model, initial_conditions, t_final, epochs, train_size, optimizer, beta, gamma, known_points,
                         num_batches=1, t_0_train=None, t_final_train=None, val_size=None,
                         selection=None, perc=1.0, treshold=float('-inf'), additional_comment='', verbose=True,
                         writer=None):
    # Check if any selection criteria on samples has been passed
    if selection is None:
        perc = 1.0

    # Move to device
    model = model.to(device)

    # Train mode
    model.train()

    # Initialize model to return
    best_model = model

    # Tensorboard writer
    if writer is None:
        writer = get_writer(t_0_train, t_final_train, initial_conditions, int(train_size * perc),
                            selection, additional_comment)

    # Initialize losses arrays
    train_losses, val_losses, min_loss = [], [], 1

    # Fetch parameters of the differential equation
    t_0, params_0 = initial_conditions[0], initial_conditions[1:]

    # Interval of points to use for training
    if t_0_train is None or t_final_train is None:
        t_0_train, t_final_train = t_0, t_final

    # Points selection
    grid = generate_grid(selection, model, initial_conditions, t_0_train, t_final_train, train_size, perc=perc)

    start_time = time.time()

    for epoch in tqdm(range(epochs), desc='Training', disable=not verbose):

        # Generate DataLoader
        batch_size = int(train_size / num_batches)
        t_dataloader = generate_dataloader(grid, t_0_train, t_final_train, batch_size, perturb=True)

        train_epoch_loss = 0.0

        for i, t in enumerate(t_dataloader, 0):
            #  Network solutions
            t = t.to(device)
            s, i, r = model.parametric_solution(t, initial_conditions)

            # Differential equation loss computation
            diff_loss = sir_loss(t, s, i, r, beta=beta, gamma=gamma)

            # Known points loss computation
            known_loss = mse_loss(known_points, model, initial_conditions)

            # Total loss is the sum of the two
            batch_loss = diff_loss + known_loss

            # Optimization
            batch_loss.backward(retain_graph=False)  # True
            optimizer.step()
            train_epoch_loss += batch_loss.item()
            optimizer.zero_grad()

        # If not specified, the validation size is the same as the train size
        if not val_size:
            val_size = train_size

        # Once every few epochs, we run validation
        if epoch % 100 == 0:
            val_epoch_loss = validate(model, initial_conditions, beta, gamma, t_final, val_size, num_batches)

        # Keep the loss function history
        train_losses.append(train_epoch_loss)
        # val_losses.append(val_epoch_loss)
        writer.add_scalar('Loss/train', train_epoch_loss, epoch)
        writer.add_scalar('Loss/val', val_epoch_loss, epoch)

        # Keep the best model (lowest loss) by using a deep copy
        if epoch > 0.8 * epochs and train_epoch_loss < min_loss:
            best_model = copy.deepcopy(model)
            min_loss = train_epoch_loss

        # If a treshold is passed, we stop training when it is reached. Notice default value is -inf
        if train_epoch_loss < treshold:
            break

    final_time = time.time()
    run_time = final_time - start_time
    return best_model, train_losses, run_time, optimizer


def validate(model, initial_conditions, beta, gamma, t_final, val_size, num_batches):
    # Grid of points to use for validation
    t_0 = initial_conditions[0]
    t = torch.linspace(t_0, t_final, val_size).reshape(-1, 1)

    batch_size = int(val_size / num_batches)
    t.requires_grad = True
    t_dataloader = DataLoader(dataset=t, batch_size=batch_size, shuffle=True)

    val_loss = 0.0

    for i, t in enumerate(t_dataloader, 0):
        t = t.to(device)

        # Network solutions
        s, i, r = model.parametric_solution(t, initial_conditions)

        # Loss computation
        val_epoch_loss = sir_loss(t, s, i, r, beta=beta, gamma=gamma)
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
