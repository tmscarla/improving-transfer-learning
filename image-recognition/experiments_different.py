from plots import *
from downloads import *
from utils import *
import datetime
from boosting.boosting import update_weights_boosting
from boosting.weighted_loss import *
from training import *
from instance_based import instance_selection


class ExperimentDifferent:
    def __init__(self, baseline, dataset, new_dataset, property, property_perc, epochs_boosting=5,
                 boosting_perc=0.25, most=True, random=False, diversity=False, alternate=False,
                 flatten=True):
        self.baseline = baseline
        self.dataset = dataset
        self.property = property
        self.property_perc = property_perc
        self.most = most
        self.diversity = diversity
        self.random = random
        self.alternate = alternate
        self.epochs_boosting = epochs_boosting
        self.boosting_perc = boosting_perc
        self.new_dataset = new_dataset
        self.flatten = flatten

    def setup_experiment(self):
        # Load baseline
        try:
            self.model_clean = load_model_from_disk(self.baseline)
        except FileNotFoundError:
            print('Baseline not found. Training...', end='')
            train_baseline(self.dataset, self.baseline, verbose=False)
            time.sleep(3)
            print('done!')
            self.model_clean = load_model_from_disk(self.baseline)

        # Load data
        (X_train, y_train), (X_test, y_test) = load_dataset(self.dataset)
        (X_train_new, y_train_new), (X_test_new, y_test_new) = load_dataset(self.new_dataset)

        return self.model_clean, (X_train, y_train), (X_test, y_test), (X_train_new, y_train_new), (
        X_test_new, y_test_new)

    ################################################ RUNS ##############################################################

    def run_jacobian(self, epochs=40, flatten=False):
        # Setup
        model, (X_train, y_train), (X_test, y_test), (X_train_new, y_train_new), (
            X_test_new, y_test_new) = self.setup_experiment()

        # Validation splitting
        (X_train_new, y_train_new), (X_valid_new, y_valid_new) = dataset_split(X_train_new, y_train_new,
                                                                               return_data='samples')

        # Image pre-processing: scale pixel values
        X_train_new_sc, X_mean, X_std = image_preprocessing(X_train_new, scale_only=False)
        X_valid_new_sc, _, _ = image_preprocessing(X_valid_new, seq_mean=X_mean, seq_std=X_std, scale_only=False)
        X_test_new_sc, _, _ = image_preprocessing(X_test_new, seq_mean=X_mean, seq_std=X_std, scale_only=False)

        n_classes = len(np.unique(y_train_new))
        criterion = nn.CrossEntropyLoss()

        # If random, I select a subset of indices as long as the length of the jacobian-based selected indices
        if self.random:
            total_len = X_train_new_sc.shape[0]
            subset_len = 3982
            selected_indices = np.random.choice(total_len, subset_len, replace=False)
            writer = SummaryWriter(
                'runs/' + self.dataset + '_random_jacobian_' + str(n_classes))
        else:

            selected_indices = instance_selection.instance_selection_no_hessian(model=model,
                                                                                X_train=X_train_new_sc,
                                                                                y_train=y_train_new,
                                                                                X_valid=X_valid_new_sc,
                                                                                y_valid=y_valid_new,
                                                                                criterion=criterion,
                                                                                flatten=flatten,
                                                                                return_influences=False,
                                                                                save_jacobian_train=False)

            writer = SummaryWriter(
                'runs/' + self.dataset + '_jacobian_' + str(n_classes) + 'len=' + str(len(selected_indices)))

        print('Total instances selected: {}/{}'.format(len(selected_indices), X_train_new_sc.shape[0]))

        X_new_subset_sc, y_new_subset = X_train_new_sc[selected_indices], y_train_new[selected_indices]

        new_subset_dataloader = get_data_loader(X_new_subset_sc, y_new_subset)
        valid_new_dl = get_data_loader(X_valid_new_sc, y_valid_new)
        test_new_dl = get_data_loader(X_test_new_sc, y_test_new)

        # Train
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        train_losses, train_accuracies, val_accuracies, \
        val_losses, _, model = train(model, new_subset_dataloader,
                                     valid_new_dl, test_new_dl, optimizer,
                                     torch.nn.CrossEntropyLoss(), device, epochs=epochs,
                                     early_stopping=False, writer=writer,
                                     save_model=False, model_path='', pbar=True,
                                     flatten=flatten, start_epoch=0)

        return train_losses, train_accuracies, val_accuracies, val_losses, model

    def run_boosting(self, lr=None, loss='cross'):
        # Setup
        model, (X_train, y_train), (X_test, y_test), (X_train_new, y_train_new), (
            X_test_new, y_test_new) = self.setup_experiment()
        n_classes = np.unique(y_train_new)

        if loss == 'exp':
            y_train_new = convert_labels(y_train_new, [0, 1], [-1, 1])
            y_test_new = convert_labels(y_test_new, [0, 1], [-1, 1])

        (X_train_new, y_train_new), (X_valid_new, y_valid_new) = dataset_split(X_train_new, y_train_new,
                                                                               return_data='samples')

        # Image pre-processing: scale pixel values
        X_train_new_sc, X_mean, X_std = image_preprocessing(X_train_new, scale_only=False)
        X_valid_new_sc, _, _ = image_preprocessing(X_valid_new, seq_mean=X_mean, seq_std=X_std, scale_only=False)
        X_test_new_sc, _, _ = image_preprocessing(X_test_new, seq_mean=X_mean, seq_std=X_std, scale_only=False)

        # TODO Mapping X to y: better come up with a smarter solution
        X_to_y = dict(zip([tuple(x.flatten()) for x in X_train_new_sc], y_train_new))

        # Noisy Dataloaders
        valid_new_dl = get_data_loader(X_valid_new_sc, y_valid_new)
        test_new_dl = get_data_loader(X_test_new_sc, y_test_new)

        X_test_sc, _, _ = image_preprocessing(X_test)
        test_clean_dl = get_data_loader(X_test_sc, y_test)
        print('Sanity check | Model performance on clean test set: {}'.format(evaluate(model,
                                                                                       test_loader=test_clean_dl,
                                                                                       device=device,
                                                                                       flatten=self.flatten)))
        print('Sanity check | Model performance on noisy test set: {}'.format(evaluate(model,
                                                                                       test_loader=test_new_dl,
                                                                                       device=device,
                                                                                       flatten=self.flatten)))

        train_losses = []
        train_accuracies = []
        valid_accuracies = []
        valid_losses = []

        learning_rate_collection = [0]
        epsilon_collection = [0]

        _, weights_boosting = init_loss(X_train_new_sc, loss=loss)
        if self.boosting_perc:
            if lr is not None:
                writer = SummaryWriter(
                    'runs/' + self.dataset + '_boosting_' + str(self.boosting_perc) + '_lr=' + str(lr) + '_' + loss
                    + str(n_classes))
            else:
                writer = SummaryWriter(
                    'runs/' + self.dataset + '_boosting_' + str(self.boosting_perc) + '_' + loss + str(n_classes))
        elif lr and not self.boosting_perc:
            writer = SummaryWriter('runs/' + self.dataset + '_boosting_lr=' + str(lr) + '_' + loss + str(n_classes))
        else:
            if self.random:
                writer = SummaryWriter('runs/' + self.dataset + '_random' + str(self.property_perc) + '_' + loss
                                       + str(n_classes))

        global_epoch = 0

        for epoch in range(self.epochs_boosting):

            if self.boosting_perc:
                points_weights = list(weights_boosting.items())
                if epoch == 0:
                    points_weights = sorted(points_weights, key=lambda x: x[1])[::-1][
                                     :int(1.0 * len(points_weights))]
                else:
                    points_weights = sorted(points_weights, key=lambda x: x[1])[::-1][
                                     :int(self.boosting_perc * len(points_weights))]
                points = [pw[0] for pw in points_weights]

                y_subset = np.array([X_to_y[p] for p in points])
                X_new_subset_sc = np.array([np.array(p).reshape((28, 28, 1)) for p in points])

            else:
                # In order to compare with the boosting method, the first epochs must be done with the full dataset
                if self.random and epoch == 0:
                    indices = get_random_subset(X_train_new_sc, y_train_new, self.property_perc,
                                                return_indices=True)

                    indices = indices[:int(1.0 * len(X_train_new_sc))]

                # Only draw the random samples at the second epoch, then use always the same
                if self.random and epoch == 1:
                    indices = get_random_subset(X_train_new_sc, y_train_new, self.property_perc,
                                                return_indices=True)

                    indices = indices[:int(self.property_perc * len(X_train_new_sc))]

                X_new_subset_sc, y_subset = X_train_new_sc[indices], y_train_new[indices]

            # print('Sanity check | Epoch {} - Training set size: {}'.format(epoch, len(X_new_subset_sc)))

            self.criterion = WeightedLoss(X=X_train_new_sc, weights_boosting=weights_boosting, loss=loss)

            # print('Sanity check | Pre-training evaluation of a single sample...')
            # rnd_idx = np.random.randint(0, len(X_train_new_sc))
            # self.evaluate_single_sample(epoch, model, X_train_new_sc, y_train_new, idx=rnd_idx,
            #                             weights_boosting=weights_boosting,
            #                             lr=lr, loss=loss)

            # Data Loader Generation
            new_subset_dataloader = get_data_loader(X_new_subset_sc, y_subset, shuffle=True)

            # Train
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00001)

            if epoch == 0:
                sub_epochs = 3
            else:
                sub_epochs = 5
            epoch_train_losses, epoch_train_accuracies, epoch_val_accuracies, \
            epoch_val_losses, global_epoch, model = train(model, new_subset_dataloader,
                                                          valid_new_dl, test_new_dl, optimizer,
                                                          self.criterion, device, epochs=sub_epochs,
                                                          early_stopping=False, writer=writer,
                                                          save_model=False, model_path='', pbar=True,
                                                          flatten=self.flatten, start_epoch=global_epoch + 1)

            train_losses.append(epoch_train_losses)
            train_accuracies.append(epoch_train_accuracies)
            valid_losses.append(epoch_val_losses)
            valid_accuracies.append(epoch_val_accuracies)

            if lr != 0.0 and not self.random:
                weights_boosting = update_weights_boosting(model,
                                                           weights_boosting,
                                                           X_train_new_sc, y_train,
                                                           learning_rate_collection=learning_rate_collection,
                                                           epsilon_collection=epsilon_collection,
                                                           device=device,
                                                           lr=lr, loss=loss, flatten=self.flatten)

            # print('Sanity check | Post-training evaluation of a single sample...')
            # self.evaluate_single_sample(epoch, model, X_train_new_sc, y_train, idx=rnd_idx,
            #                             weights_boosting=weights_boosting,
            #                             lr=lr, loss=loss)

            # Evaluation
            test_epoch_acc = evaluate(model, test_new_dl, device, flatten=self.flatten)

            print('---------------------------------------------------------------------------------------------------')
            print('Boosting epoch {:3d} | Train loss: {:5f} | Train acc: {:5f} | Val acc: {:5f} | Val loss: {:5f}'
                  ' | Test acc: {:5f} \nBoosting parameters: Epsilon: {} | Lambda: {}'.format(epoch,
                                                                                              epoch_train_losses[-1],
                                                                                              epoch_train_accuracies[
                                                                                                  -1],
                                                                                              epoch_val_accuracies[-1],
                                                                                              epoch_val_losses[-1],
                                                                                              test_epoch_acc,
                                                                                              epsilon_collection[-1],
                                                                                              learning_rate_collection[
                                                                                                  -1]))
            print('---------------------------------------------------------------------------------------------------')

        final_test_acc = evaluate(model, test_new_dl, device, flatten=self.flatten)
        print('Your final test accuracy is {}'.format(final_test_acc))

        # generate_boosting_plots(self.epochs_boosting, train_accuracies, train_losses, valid_accuracies, weights,
        #                        learning_rate_collection, epsilon_collection, lr=lr)

        return train_accuracies, valid_accuracies


    def run_usps(self):
        try:
            model = load_model_from_disk(self.baseline)
        except RuntimeError:
            train_baseline(self.dataset, self.baseline, classes=self.classes, verbose=False)

        (X_train, y_train), (X_test, y_test) = load_USPS()

        (X_train, y_train), (X_valid, y_valid) = dataset_split(X_train, y_train, return_data='samples')
        train_loader = get_data_loader(X_train, y_train, shuffle=False)
        val_loader = get_data_loader(X_valid, y_valid, shuffle=False)
        test_loader = get_data_loader(X_test, y_test, shuffle=False)

        # Model path and Writer
        model_path, writer = self.get_model_path_and_writer()

        # Optimizer and criterion
        optimizer = self.get_optimizer(model)
        criterion = nn.CrossEntropyLoss()

        acc = evaluate(model, test_loader, device, flatten=True)
        print("Accuracy before training = {}".format(acc), flush=True)

        train(model, train_loader, val_loader, test_loader, optimizer,
              criterion, device, writer, model_path=model_path, pbar=True, flatten=True)

        # Evaluation
        acc = evaluate(model, test_loader, device, flatten=True)
        print("Accuracy after training = {}".format(acc))

    def run_recompute(self, total_epochs=50, recompute_epochs=1):
        # Setup
        model, (X_train, y_train), (X_test, y_test), (X_train_new, y_train_new), (
        X_test_new, y_test_new) = self.setup_experiment()
        test_loader_new = get_data_loader(X_test_new, y_test_new, shuffle=False)
        print('Starting accuracy: {:.6f}'.format(evaluate(model, test_loader_new, device, flatten=True)))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        model_path, writer = self.get_model_path_and_writer(other='_recompute')

        flatten = False
        use_lambda = False
        l = 0.001

        # # TODO SPLIT ONLY ONCE TRAIN AND VALID BEFORE SUBSAMPLING
        # (X_train_noisy, y_train), (X_valid_noisy, y_valid) = self.train_validation_split(X_train_noisy, y_train,
        #                                                                                  data_loader=False)

        if self.dataset == 'MNIST':
            flatten = True

        unique_labels = list(np.unique(y_train))
        old_values = np.zeros(len(X_train_new))
        old_indices_sub = [0]

        i = 0
        while i < total_epochs:
            model.to(device)

            # Generate specific subset
            if self.random:
                X_train_new_sub, y_train_sub = get_random_subset(X_train_new, y_train_new, self.property_perc)
            else:
                if use_lambda:
                    values = compute_samples_property(model, X_train_new, y_train_new, unique_labels,
                                                      self.property, indices=False, flatten=flatten)
                    values = np.array(values)
                    values = np.interp(values, (values.min(), values.max()), (0, 1))
                    new_values = old_values - l * (old_values - values)
                    n_samples = int(len(new_values) * self.property_perc)
                    indices = np.argsort(new_values)
                    if self.most: indices = indices[::-1]
                    indices_sub = indices[:n_samples]
                    print('Samples changed: {}'.format(len(np.setdiff1d(indices_sub, old_indices_sub))))
                    old_values = new_values
                    old_indices_sub = indices_sub
                    X_train_new_sub, y_train_sub = X_train_new[indices_sub], y_train_new[indices_sub]
                else:
                    if self.alternate and i % 2 == 0:
                        X_train_new_sub, y_train_sub = get_random_subset(X_train_new, y_train_new, self.property_perc)
                    else:
                        X_train_new_sub, y_train_sub = get_samples_by_property(model, X_train_new, y_train_new,
                                                                                 self.property_perc, self.most,
                                                                                 prop=self.property, diversity=False,
                                                                                 flatten=flatten)
            # Create DataLoader objects
            # TODO THIS IS NORMAL MODE
            train_loader_new_sub, val_loader_new_sub = dataset_split(X_train_new_sub, y_train_sub,
                                                                         return_data='data_loader')
            # TODO THIS IS USING THE SAME VALIDATION
            # train_loader_noisy_sub = get_data_loader(X_train_noisy_sub, y_train_sub)
            # val_loader_noisy_sub = get_data_loader(X_valid_noisy, y_valid)

            # Training
            train(model, train_loader_new_sub, val_loader_new_sub, test_loader_new, optimizer,
                  torch.nn.CrossEntropyLoss(), device, epochs=recompute_epochs, early_stopping=False,
                  writer=writer, start_epoch=i, save_model=False, model_path=model_path, pbar=True,
                  flatten=flatten)

            i += recompute_epochs

    def run(self):
        # Setup
        model, (X_train, y_train), (X_test, y_test), (X_train_new, y_train_new), (
        X_test_new, y_test_new) = self.setup_experiment()

        # Scale datasets
        X_train, X_mean, X_std = image_preprocessing(X_train, scale_only=False)
        X_test, _, _ = image_preprocessing(X_test, seq_mean=X_mean, seq_std=X_std, scale_only=False)

        # Image pre-processing: scale pixel values
        X_train_new, X_mean, X_std = image_preprocessing(X_train_new, scale_only=False)
        X_test_new, _, _ = image_preprocessing(X_test_new, seq_mean=X_mean, seq_std=X_std, scale_only=False)

        if self.dataset == 'MNIST':
            flatten = True
        else:
            flatten = False

        # Generate specific subset
        if self.random:
            X_train_new_sub, y_train_sub = get_random_subset(X_train_new, y_train_new, self.property_perc)
        else:
            X_train_new_sub, y_train_sub = get_samples_by_property(model, X_train_new, y_train_new,
                                                                     self.property_perc, self.most,
                                                                     prop=self.property,
                                                                     flatten=flatten)

        # Create DataLoader objects
        train_loader_new_sub, val_loader_new_sub = dataset_split(X_train_new_sub, y_train_sub,
                                                                     return_data='data_loader')

        test_loader_new = get_data_loader(X_test_new, y_test_new, shuffle=False)
        test_loader = get_data_loader(X_test, y_test, shuffle=False)

        # Evaluation
        acc_clean = evaluate(model, test_loader, device, flatten=flatten)
        acc_noisy = evaluate(model, test_loader_new, device, flatten=flatten)
        print('Starting... Your accuracy on (x_test_clean) = %.3f' % acc_clean)
        print('Starting... Your accuracy on (x_test_noisy) = %.3f' % acc_noisy)

        # Model path and Writer
        model_path, writer = self.get_model_path_and_writer()

        # Optimizer and criterion
        optimizer = self.get_optimizer(model)
        criterion = nn.CrossEntropyLoss()

        # Training
        train(model, train_loader_new_sub, val_loader_new_sub, test_loader_new, optimizer,
              criterion, device, writer, model_path=model_path, pbar=True, flatten=flatten, epochs=100)

        # early_stopping(model, train_loader_noisy_sub, val_loader_noisy_sub, test_loader_noisy, optimizer, device,
        #                model_path, writer)

        # Evaluation
        acc_noisy = evaluate(model, test_loader_new, device, flatten=flatten)
        acc_clean = evaluate(model, test_loader, device, flatten=flatten)
        print('Your accuracy on (x_test_clean) = %.3f' % acc_clean)
        print('Your accuracy on (x_test_noisy) = %.3f' % acc_noisy)

    def get_model_path_and_writer(self, other=''):
        """
        Get the path to save the model and compose the name of the experiment, according to its features
        :param other: optional string to be appended to the end of the name
        :return:
        """
        if self.random:
            model_path = os.path.join(RETRAINED_DIR,
                                      self.dataset + '_random' + str(self.property_perc))
            writer = SummaryWriter('runs/' + self.dataset + '_random' + str(self.property_perc))
            return model_path, writer

        if self.most:
            model_path = os.path.join(RETRAINED_DIR,
                                      self.dataset + '_top' + str(self.property_perc) + '_' + self.property)
            writer = SummaryWriter('runs/' + self.dataset + '_top' + str(self.property_perc) + '_' + self.property)
        else:
            model_path = os.path.join(RETRAINED_DIR,
                                      self.dataset + '_bottom' + str(self.property_perc) + '_' + self.property)
            writer = SummaryWriter('runs/' + self.dataset + '_bottom' + str(self.property_perc) + '_' + self.property)

        if not os.path.exists(os.path.join(MODELS_DIR, model_path)):
            os.makedirs(os.path.join(MODELS_DIR, model_path))

        model_path = os.path.join(model_path, datetime.datetime.now().strftime('%d-%m-%Y_%H:%M:%S') + '.pt')

        return model_path, writer

    def get_optimizer(self, net, weight_decay=0):
        if self.dataset == 'CIFAR_10':
            # freeze fully connected layers
            # for name, param in net.named_parameters():
            #     if 'fc' in name:
            #         param.requires_grad = False
            #     else:
            #         param.requires_grad = True
            # # passing only those parameters that explicitly requires grad
            # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.000001, weight_decay=weight_decay)
            optimizer = torch.optim.Adam(net.parameters(), lr=0.001, weight_decay=weight_decay)
        elif self.dataset == 'CIFAR_100':
            # freeze all but first two layers
            for name, param in net.named_parameters():
                if name[4] in ['3', '4', '5', '6', '7', '8', '9']:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            # passing only those parameters that explicitly requires grad
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.000001,
                                         weight_decay=weight_decay)
        else:
            optimizer = torch.optim.Adam(net.parameters())
        return optimizer


if __name__ == '__main__':
    ed = ExperimentDifferent(baseline='FFNetBaseline', dataset='MNIST', new_dataset='USPS', property='cross_entropy',
                             most=True, property_perc=0.50, random=True)
    #ed.run_boosting(lr=10)
    #ed.run_recompute(recompute_epochs=5, total_epochs=50)
    ed.run()
    #ed.run_jacobian(flatten=True, epochs=50)
