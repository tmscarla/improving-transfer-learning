import torch
from downloads import load_MNIST
from instance_based.models import InstanceMNISTNet
import numpy as np
from torchsummary import summary
from tqdm import tqdm
from scipy import sparse as sps
import similaripy as sim
from scipy.sparse.linalg import inv
from constants import device


def hessian_helper(loss_grad, model):
    cnt = 0
    for g in loss_grad:
        g_vector = g.contiguous().view(-1) if cnt == 0 else torch.cat([g_vector, g.contiguous().view(-1)])
        cnt = 1
    l = g_vector.size(0)
    hessian = torch.zeros(l, l)
    hessian = hessian.to(device)
    for idx in range(l):
        grad2rd = torch.autograd.grad(g_vector[idx], model.parameters(), create_graph=True)
        cnt = 0
        for g in grad2rd:
            g2 = g.contiguous().view(-1) if cnt == 0 else torch.cat([g2, g.contiguous().view(-1)])
            cnt = 1
        hessian[idx] = g2
    return hessian


def hessian_helper_sparse(grads, params, mode='raw', threshold_value=0):
    start_row, start_col = 0, 0
    rows, cols, data = [], [], []

    for i, (grad, p) in enumerate(zip(grads, params)):
        grad = grad.reshape(-1)
        d = len(grad)

        for j, g in enumerate(grad):
            grad2 = torch.autograd.grad(g, p, create_graph=True)[0].view(-1)

            if mode == 'treshold':
                for k, g2 in enumerate(grad2):
                    if abs(g2) >= threshold_value:
                        rows.append(start_row + j)
                        cols.append(start_col + k)
                        data.append(g2.item())
            elif mode == 'diagonal':
                for k, g2 in enumerate(grad2):
                    if k == j:
                        rows.append(start_row + j)
                        cols.append(start_col + k)
                        data.append(g2.item())
            else:
                rows += [start_row + j for _ in range(len(grad2))]
                cols += [start_col + k for k in range(len(grad2))]
                data += grad2.tolist()

        start_row += d
        start_col += d

    return np.array(rows), np.array(cols), np.array(data)


def compute_treshold(model, X_train, y_train, criterion, add_channel, percentile=30):
    for i, train_sample in enumerate(tqdm(X_train, desc='Compute treshold')):
        for name, param in model.state_dict().items():
            if param.dtype == torch.float32:
                param.requires_grad = True

        if add_channel:
            train_sample = np.expand_dims(train_sample, axis=0)

        train_sample = torch.Tensor(train_sample)
        label = y_train[i]
        label = torch.LongTensor([label])
        train_sample, model, label = train_sample.to(device), model.to(device), label.to(device)
        output = model(train_sample)
        loss = criterion(output, label)

        grads, params = torch.autograd.grad(loss, model.parameters(), create_graph=True), model.parameters()
        for grad, p in zip(grads, params):
            grad = grad.reshape(-1)
            g2_list = []

            for g in grad:
                g2 = torch.autograd.grad(g, p, create_graph=True)[0].view(-1)
                g2_list.append(g2)

            if i == 0:
                g2_avg = torch.cat(g2_list)
            else:
                g2_avg += torch.cat(g2_list)
            break

    g2_avg /= len(X_train)
    g2_avg = torch.abs(g2_avg)
    return np.percentile(g2_avg.detach().numpy(), percentile)


def compute_hessian_treshold(model, X_train, y_train, criterion, add_channel, percentile=30, flatten=False):
    # Compute treshold
    treshold = compute_treshold(model, X_train, y_train, criterion, add_channel, percentile)
    hessian_size = 0

    for name, param in model.state_dict().items():
        if param.dtype == torch.float32:
            param.requires_grad = True
            hessian_size += np.prod(param.shape)

    # For each parameter compute a sparse block of the hessian
    start = 0
    rows, cols, data = [], [], []
    for param in tqdm(model.parameters(), desc='Hessian sparse treshold'):
        rows_p, cols_p, data_p = hessian_helper_treshold(model, X_train, y_train, criterion,
                                                         add_channel, param, treshold, start, flatten)
        start += np.prod(param.shape)
        rows, cols, data = rows + rows_p, cols + cols_p, data + data_p

    hessian = sps.csc_matrix((data, (rows, cols)), shape=(hessian_size, hessian_size))
    return hessian


def hessian_helper_treshold(model, X_train, y_train, criterion, add_channel, param, treshold, start, flatten=False):
    l = np.prod(param.shape)
    hessian_block = torch.zeros(l, l)

    # Build a dense hessian block for param
    for i, train_sample in enumerate(X_train):
        if add_channel:
            train_sample = np.expand_dims(train_sample, axis=0)

        train_sample = torch.Tensor(train_sample)
        label = y_train[i]
        label = torch.LongTensor([label])
        train_sample, model, label = train_sample.to(device), model.to(device), label.to(device)

        if flatten:
            train_sample = train_sample.view(train_sample.shape[0], -1)

        output = model(train_sample)
        loss = criterion(output, label)

        grad = torch.autograd.grad(loss, param, create_graph=True)[0]
        grad = grad.reshape(-1)
        hessian_block_i = torch.zeros(l, l)

        for j, g in enumerate(grad):
            g2 = torch.autograd.grad(g, param, create_graph=True)[0].view(-1)
            hessian_block_i[j] = g2

        hessian_block = torch.add(hessian_block, hessian_block_i)

    # Average all the samples
    hessian_block /= len(X_train)

    # Make it sparse according to the treshold
    rows, cols, data = [], [], []

    for i in range(hessian_block.size(0)):
        for j in range(hessian_block.size(1)):
            if abs(hessian_block[i][j]) > treshold:
                rows.append(start + i)
                cols.append(start + j)
                data.append(hessian_block[i][j].item())

    return rows, cols, data


def compute_hessian(model, X_train, y_train, criterion, sparse=True, treshold=False, add_channel=True, flatten=False):
    num_params = sum(p.numel() for p in model.parameters())

    if sparse:
        if treshold:
            return compute_hessian_treshold(model, X_train, y_train, criterion, add_channel, percentile=30,
                                            flatten=flatten)
        rows, cols, data = np.array([]), np.array([]), np.array([])
        desc = "Hessian matrix sparse"
    else:
        hessian_matrix = torch.zeros((num_params, num_params))
        hessian_matrix = hessian_matrix.to(device)
        desc = "Hessian matrix"

    for i, train_sample in enumerate(tqdm(X_train, desc=desc)):
        for name, param in model.state_dict().items():
            if param.dtype == torch.float32:
                param.requires_grad = True

        if add_channel:
            train_sample = np.expand_dims(train_sample, axis=0)

        train_sample = torch.Tensor(train_sample)
        label = y_train[i]
        label = torch.LongTensor([label])
        train_sample, model, label = train_sample.to(device), model.to(device), label.to(device)

        if flatten:
            train_sample = train_sample.view(train_sample.shape[0], -1)

        output = model(train_sample)
        loss = criterion(output, label)

        if sparse:
            grads, params = torch.autograd.grad(loss, model.parameters(), create_graph=True), model.parameters()
            rows_i, cols_i, data_i = hessian_helper_sparse(grads, params, mode='raw')
            if i == 0:
                rows, cols, data = rows_i, cols_i, data_i
            else:
                data = np.add(data, data_i)
        else:
            hessian_matrix = torch.add(hessian_matrix,
                                       hessian_helper(torch.autograd.grad(loss, model.parameters(), create_graph=True),
                                                      model))

    if sparse:
        data = data / num_params
        hessian_matrix = sps.csr_matrix((data, (rows, cols)))
    else:
        hessian_matrix = hessian_matrix / num_params

    return hessian_matrix


def compute_hvp(X_train, y_train, selected_samples_indices, jacobian, model, criterion, approximation_grade,
                flatten):
    if approximation_grade == 0:
        return jacobian
    else:
        sample = X_train[selected_samples_indices[approximation_grade - 1]]

        model.zero_grad()
        train_sample = np.expand_dims(sample, axis=0)
        train_sample = torch.Tensor(train_sample)
        label = y_train[selected_samples_indices[approximation_grade - 1]]
        label = torch.LongTensor([label])
        train_sample, model, label = train_sample.to(device), model.to(device), label.to(device)

        if flatten:
            train_sample = train_sample.view(train_sample.shape[0], -1)

        output = model(train_sample)
        loss = criterion(output, label)

        hessian = hessian_helper(torch.autograd.grad(loss, model.parameters(), create_graph=True),
                                 model)

        identity = torch.eye(hessian.shape[0])

        return torch.add(jacobian,
                         torch.matmul((identity - hessian), compute_hvp(X_train, y_train, selected_samples_indices,
                                                                        jacobian, model, criterion,
                                                                        approximation_grade - 1,
                                                                        flatten)))


def instance_selection_hvp(model, X_train, y_train, X_valid, y_valid,
                           criterion, flatten=False, approximation_grade=3):
    hvp_collector = []
    for j, valid_sample in enumerate(tqdm(X_valid, desc='HVP computation')):
        model.zero_grad()
        valid_sample = np.expand_dims(valid_sample, axis=0)
        valid_sample = torch.Tensor(valid_sample)
        label = y_valid[j]
        label = torch.LongTensor([label])
        valid_sample, model, label = valid_sample.to(device), model.to(device), label.to(device)

        if flatten:
            valid_sample = valid_sample.view(valid_sample.shape[0], -1)

        output = model(valid_sample)
        loss = criterion(output, label)
        loss.backward()

        jacobian_j = []
        for param in model.parameters():
            jacobian_j.append(param.grad.view(-1))
        jacobian_j = torch.cat(jacobian_j)

        selected_samples_indices = np.random.choice(len(X_train), approximation_grade, replace=False)

        hvp = compute_hvp(model=model, X_train=X_train, y_train=y_train,
                          selected_samples_indices=selected_samples_indices,
                          jacobian=jacobian_j, criterion=criterion,
                          approximation_grade=approximation_grade, flatten=flatten)

        hvp_collector.append(hvp)

    selected_indices = []
    for i, train_sample in enumerate(tqdm(X_train, desc='Instance selection')):
        model.zero_grad()
        train_sample = np.expand_dims(train_sample, axis=0)
        train_sample = torch.Tensor(train_sample)
        label = y_train[i]
        label = torch.LongTensor([label])
        train_sample, model, label = train_sample.to(device), model.to(device), label.to(device)

        if flatten:
            train_sample = train_sample.view(train_sample.shape[0], -1)

        output = model(train_sample)
        loss = criterion(output, label)
        loss.backward()

        jacobian_i = []
        for idx, param in enumerate(model.parameters()):
            jacobian_i.append(param.grad.view(-1))
        jacobian_i = torch.cat(jacobian_i)

        j_loss = 0

        for hvp in hvp_collector:
            j_loss = j_loss + torch.matmul(jacobian_i, hvp) * (-1)

        if j_loss <= 0:
            selected_indices.append(i)

    return selected_indices


def instance_selection(model, X_train, y_train, X_valid, y_valid, criterion,
                       sparse=True, add_channel=True, flatten=False, treshold=False, return_influences=False):
    hessian_matrix = compute_hessian(model, X_train, y_train, criterion, sparse=sparse, add_channel=add_channel,
                                     flatten=flatten, treshold=treshold)

    if sparse:
        print('Hessian matrix sparsity: {:.2f}%'.format(hessian_matrix.nnz / (hessian_matrix.shape[0] ** 2) * 100))
        hessian_matrix_inv = inv(hessian_matrix)
    else:
        hessian_matrix_inv = torch.inverse(hessian_matrix)

    selected_indices = []
    influences = []
    for i, train_sample in enumerate(tqdm(X_train, desc='Instance selection')):
        model.zero_grad()
        train_sample = np.expand_dims(train_sample, axis=0)
        train_sample = torch.Tensor(train_sample)
        label = y_train[i]
        label = torch.LongTensor([label])
        train_sample, model, label = train_sample.to(device), model.to(device), label.to(device)

        if flatten:
            train_sample = train_sample.view(train_sample.shape[0], -1)

        output = model(train_sample)
        loss = criterion(output, label)
        loss.backward()

        jacobian_i = []
        for idx, param in enumerate(model.parameters()):
            jacobian_i.append(param.grad.view(-1))
        jacobian_i = torch.cat(jacobian_i)

        if sparse:
            jacobian_i = sps.csc_matrix(jacobian_i.tolist()).transpose(copy=False)
            intermediate = sim.dot_product(hessian_matrix_inv, jacobian_i, verbose=False)
        else:
            jacobian_i = jacobian_i.to(device)
            intermediate = torch.matmul(hessian_matrix_inv, jacobian_i)

        j_loss = 0
        for j, valid_sample in enumerate(X_valid):
            model.zero_grad()
            valid_sample = np.expand_dims(valid_sample, axis=0)
            valid_sample = torch.Tensor(valid_sample)
            label = y_valid[j]
            label = torch.LongTensor([label])
            valid_sample, model, label = valid_sample.to(device), model.to(device), label.to(device)

            if flatten:
                valid_sample = valid_sample.view(valid_sample.shape[0], -1)

            output = model(valid_sample)
            loss = criterion(output, label)
            loss.backward()

            jacobian_j = []
            for param in model.parameters():
                jacobian_j.append(param.grad.view(-1))
            jacobian_j = torch.cat(jacobian_j)

            if sparse:
                jacobian_j = sps.csc_matrix(jacobian_j.tolist())
                j_loss += sim.dot_product(jacobian_j * (-1), intermediate, verbose=False).data[0]
            else:
                jacobian_j = jacobian_j.to(device)
                j_loss += torch.matmul((jacobian_j * (-1)), intermediate)

        influences.append(j_loss)
        if j_loss <= 0:
            selected_indices.append(i)

    return influences if return_influences else selected_indices


def instance_selection_no_hessian(model, X_train, y_train, X_valid, y_valid, criterion, flatten=False,
                                  return_influences=False, save_jacobian_train=False):
    selected_indices = []
    influences = []
    jacobian_train = []
    jacobian_valid = []

    for i, train_sample in enumerate(tqdm(X_train, desc='Jacobian train')):
        j_loss = 0
        model.zero_grad()

        # If it is an image, put channel first
        if len(train_sample.shape) == 3:
            train_sample = np.moveaxis(train_sample, source=-1, destination=0)

        train_sample = np.expand_dims(train_sample, axis=0)
        train_sample = torch.Tensor(train_sample)
        label = y_train[i]
        label = torch.LongTensor([label])
        train_sample, model, label = train_sample.to(device), model.to(device), label.to(device)

        if flatten:
            train_sample = train_sample.view(train_sample.shape[0], -1)

        output = model(train_sample)
        loss = criterion(output, label)
        loss.backward()

        jacobian_i = []
        for idx, param in enumerate(model.parameters()):
            jacobian_i.append(param.grad.view(-1))
        jacobian_i = torch.cat(jacobian_i)

        if save_jacobian_train:
            jacobian_train.append(jacobian_i)
        else:
            # This branch is taken in the case I won't store the jacobians of the training set
            # in memory, so for each of them I will compute directly the influence loss
            for j, valid_sample in enumerate(X_valid):
                model.zero_grad()

                # If it is an image, put channel first
                if len(valid_sample.shape) == 3:
                    valid_sample = np.moveaxis(valid_sample, source=-1, destination=0)

                valid_sample = np.expand_dims(valid_sample, axis=0)
                valid_sample = torch.Tensor(valid_sample)
                label = y_valid[j]
                label = torch.LongTensor([label])
                valid_sample, model, label = valid_sample.to(device), model.to(device), label.to(device)

                if flatten:
                    valid_sample = valid_sample.view(valid_sample.shape[0], -1)

                output = model(valid_sample)
                loss = criterion(output, label)
                loss.backward()

                jacobian_j = []
                for param in model.parameters():
                    jacobian_j.append(param.grad.view(-1))
                jacobian_j = torch.cat(jacobian_j)

                j_loss += torch.matmul(jacobian_i * (-1), jacobian_j)

            influences.append(j_loss)
            if j_loss <= 0:
                selected_indices.append(i)

    if not save_jacobian_train:
        return influences if return_influences else selected_indices

    # This branch is taken in the case I store the jacobians of the training set in memory
    for j, valid_sample in enumerate(tqdm(X_valid, desc='Jacobian valid')):
        model.zero_grad()

        # If it is an image, put channel first
        if len(valid_sample.shape) == 3:
            valid_sample = np.moveaxis(valid_sample, source=-1, destination=0)

        valid_sample = np.expand_dims(valid_sample, axis=0)
        valid_sample = torch.Tensor(valid_sample)
        label = y_valid[j]
        label = torch.LongTensor([label])
        valid_sample, model, label = valid_sample.to(device), model.to(device), label.to(device)

        if flatten:
            valid_sample = valid_sample.view(valid_sample.shape[0], -1)

        output = model(valid_sample)
        loss = criterion(output, label)
        loss.backward()

        jacobian_j = []
        for param in model.parameters():
            jacobian_j.append(param.grad.view(-1))
        jacobian_j = torch.cat(jacobian_j)

        jacobian_valid.append(jacobian_j)

    for i in tqdm(range(len(X_train)), desc='Instance selection without hessian'):
        j_loss = 0
        for j in range(len(X_valid)):
            j_loss += torch.matmul(jacobian_train[i] * (-1), jacobian_valid[j])
        influences.append(j_loss)
        if j_loss <= 0:
            selected_indices.append(i)

    return influences if return_influences else selected_indices


def instance_selection_train_derivatives(model, X, y, criterion, flatten=False):

    jacobians_norms = []

    for i, sample in enumerate(tqdm(X, desc='Train derivatives squared')):
        model.zero_grad()
        sample = np.expand_dims(sample, axis=0)
        sample = torch.Tensor(sample)
        label = y[i]
        label = torch.LongTensor([label])
        sample, model, label = sample.to(device), model.to(device), label.to(device)

        if flatten:
            sample = sample.view(sample.shape[0], -1)

        output = model(sample)
        loss = criterion(output, label)
        loss.backward()

        jacobian_i = []
        for param in model.parameters():
            jacobian_i.append(param.grad.view(-1))
        jacobian_i = torch.norm(torch.cat(jacobian_i))

        jacobians_norms.append(jacobian_i)

    return jacobians_norms

