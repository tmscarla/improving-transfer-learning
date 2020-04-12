from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from boosting.boosting import *
from toy_utils import *
from toy_models import *
import toy_models
from training import *
import copy
from sampler import *
import numpy as np
from animator import Animator


class ToyExperiment:
    def __init__(self, n_features, hidden_dim, n_classes, n_samples, property='entropy', property_perc=0.2,
                 theta=45, N_EPOCHS=20, alternate=False, most=True, convert_y=False):
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.property = property
        self.property_perc = property_perc
        self.theta = theta
        self.N_EPOCHS = N_EPOCHS
        self.alternate = alternate
        self.most = most
        self.setup_clean(convert_y=convert_y)
        self.setup_noisy()

    def setup_clean(self, convert_y=True):
        """
        Setup a clean setting a train a model on it. Numpy arrays of samples such as X_train, X_test
        and X_valid are not scaled. Dataloaders object instead are initialized with scaled arrays.
        :param convert_y: Convert labels from [0,1] to [-1,1]
        """
        # Generate points
        self.X, self.y = make_classification(n_samples=self.n_samples, n_features=self.n_features,
                                             n_informative=self.n_features,
                                             n_redundant=0, n_classes=self.n_classes, n_clusters_per_class=1)
        if convert_y:
            self.y = convert_labels(self.y, old_labels=[0, 1], new_labels=[-1, 1])

        # Train-Test and Train-Validation
        self.train_idx_, self.test_idx = dataset_split(
            self.X, self.y, perc=TEST_PERCENTAGE, return_data='indices')
        self.X_train, self.X_test = self.X[self.train_idx_], self.X[self.test_idx]
        self.y_train, self.y_test = self.y[self.train_idx_], self.y[self.test_idx]

        self.train_idx, self.valid_idx = dataset_split(
            self.X_train, self.y_train, perc=VALIDATION_PERCENTAGE, return_data='indices')
        self.X_train, self.X_valid = self.X_train[self.train_idx], self.X_train[self.valid_idx]
        self.y_train, self.y_valid = self.y_train[self.train_idx], self.y_train[self.valid_idx]

        # Scaling of X
        ss = StandardScaler()
        X_train_scaled = ss.fit_transform(self.X_train)
        X_valid_scaled = ss.transform(self.X_valid)
        X_test_scaled = ss.transform(self.X_test)

        # Clean Dataloaders
        self.train_dl = get_data_loader(X_train_scaled, self.y_train)
        self.valid_dl = get_data_loader(X_valid_scaled, self.y_valid)
        self.test_dl = get_data_loader(X_test_scaled, self.y_test)

        print('Training clean model...', end='')
        if convert_y:
            self.model_clean = toy_models.FFSimpleNet(input_dim=self.n_features, output_dim=1, activation='tanh')
            self.criterion, weights_boosting = init_exponential_loss(X=X_train_scaled)
        else:
            if self.n_classes <= 1:
                self.model_clean = toy_models.FFSimpleNet(input_dim=self.n_features, output_dim=1,
                                                          activation='sigmoid')
                criterion = nn.BCELoss()
            else:
                self.model_clean = toy_models.FFSimpleNet(input_dim=self.n_features, hidden_dim=self.hidden_dim,
                                                          output_dim=self.n_classes,
                                                          activation='softmax')
                self.criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.model_clean.parameters(), lr=0.01)
        self.train_losses_clean, self.test_accs_clean, self.val_accs_clean, self.val_losses_clean, epoch, model = train(
            model=self.model_clean, train_loader=self.train_dl, val_loader=self.valid_dl, test_loader=self.test_dl,
            criterion=self.criterion, optimizer=optimizer, device=device, verbose=False, flatten=True, early_stopping=False,
            epochs=self.N_EPOCHS)
        print('done!')

    def setup_noisy(self):
        # Rotation
        X_s = StandardScaler().fit_transform(self.X)
        self.X_noisy = rotate(X_s, theta=self.theta)

        # Training, Validation and Test
        self.X_train_noisy, self.X_test_noisy = self.X_noisy[self.train_idx_], self.X_noisy[self.test_idx]
        self.X_train_noisy, self.X_valid_noisy = self.X_train_noisy[self.train_idx], self.X_train_noisy[self.valid_idx]

        # Scaling of X noisy
        ss = StandardScaler()
        X_train_noisy_sc = ss.fit_transform(self.X_train_noisy)
        X_test_noisy_sc = ss.transform(self.X_test_noisy)
        X_valid_noisy_sc = ss.transform(self.X_valid_noisy)

        # Noisy Dataloaders
        self.train_noisy_dl = get_data_loader(X_train_noisy_sc, self.y_train)
        self.test_noisy_dl = get_data_loader(X_test_noisy_sc, self.y_test)
        self.valid_noisy_dl = get_data_loader(X_valid_noisy_sc, self.y_valid)

    ##################### RUNS ##############################################################################

    def run_bagging(self, random=False):

        model_clean = self.model_clean

        noisy_test_accuracy = evaluate(model_clean, self.test_noisy_dl, device)

        print('Initial test accuracy of baseline: {}'.format(noisy_test_accuracy))

        print('Training entropy model...\n', end='')
        model_entropy = FFSimpleNet(input_dim=self.n_features, output_dim=1, activation='sigmoid')
        optimizer = torch.optim.SGD(model_entropy.parameters(), lr=0.01)
        criterion = nn.BCELoss()

        # Sampling
        if not random:
            X_noisy_subset, y_subset = get_samples_by_property(model_entropy, self.X_train_noisy, self.y_train,
                                                               self.property_perc, most=self.most,
                                                               prop='entropy')
        else:
            X_noisy_subset, y_subset = get_random_subset(self.X_train_noisy, self.y_train, self.property_perc)

        # Scaling
        ss = StandardScaler()
        ss.fit(self.X_train_noisy)
        ss.transform(X_noisy_subset)

        # Data Loader generation
        noisy_subset_dataloader = get_data_loader(X_noisy_subset, y_subset)

        train_losses = []
        train_accuracies = []
        test_bagging_accuracies = []

        for epoch in range(self.N_EPOCHS):
            train_epoch_loss, train_epoch_acc = train(model_entropy, noisy_subset_dataloader, optimizer, criterion,
                                                      device)
            train_losses.append(train_epoch_loss)
            train_accuracies.append(train_epoch_acc)
            test_bagging_epoch_acc = evaluate_bagging(model_clean, model_entropy, self.test_noisy_dl, device)
            test_bagging_accuracies.append(test_bagging_epoch_acc)
            print('Epoch {} | Train acc: {:.3f} | Train loss: {:.3f} | Bagging test acc: {:.3f}'.format(epoch,
                                                                                                        train_epoch_acc,
                                                                                                        train_epoch_loss,
                                                                                                        test_bagging_epoch_acc))
        return train_losses, train_accuracies, test_bagging_accuracies

    def toy_run_boosting(self, boosting_epochs=5, random=False, lr=0.01, boosting_perc=0.7, mode='normal'):
        model = copy.deepcopy(self.model_clean)

        train_losses = []
        train_accuracies = []
        val_accuracies = []

        ss = StandardScaler()
        X_noisy_train_scaled = ss.fit_transform(self.X_train_noisy)
        X_noisy_test_scaled = ss.transform(self.X_test_noisy)
        noisy_test_dataloader = get_data_loader(X_noisy_test_scaled, self.y_test)

        X_to_y = dict(zip([tuple(x.flatten()) for x in X_noisy_train_scaled], self.y_train))

        to_monitor = np.random.randint(0, len(X_noisy_train_scaled))
        to_monitor_weights = []
        to_monitor_predictions = []

        weights_boosting = dict()
        for sample in X_noisy_train_scaled:
            weights_boosting[tuple(sample)] = 1 / len(X_noisy_train_scaled)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        criterion = WeightedLoss(X=X_noisy_train_scaled, weights_boosting=weights_boosting)

        for epoch in range(boosting_epochs):

            if random:
                indices = get_random_subset(self.X_train_noisy, self.y_train, self.property_perc,
                                            return_indices=True)
                X_noisy_subset = self.X_train_noisy[indices]
                y_subset = self.y_train[indices]
            else:
                points_weights = list(weights_boosting.items())

                # First epoch is done with the whole dataset
                if epoch == 0:
                    points_weights = sorted(points_weights, key=lambda x: x[1])[::-1][
                                     :int(1.0 * len(points_weights))]
                else:
                    points_weights = sorted(points_weights, key=lambda x: x[1])[::-1][
                                     :int(boosting_perc * len(points_weights))]
                points = [pw[0] for pw in points_weights]

                X_noisy_subset = np.array([np.array(p) for p in points])
                y_subset = np.array([X_to_y[p] for p in points])

            print('Training set size at epoch {} : {}'.format(epoch, len(X_noisy_subset)))
            print('Sanity check | Pre-training evaluation of a single sample...')
            rnd_idx = np.random.randint(0, len(X_noisy_train_scaled))
            self.evaluate_single_sample(epoch, model, X_noisy_train_scaled, self.y_train, idx=rnd_idx,
                                        weights_boosting=weights_boosting,
                                        lr=lr, loss=criterion)

            # Rescale the input
            X_noisy_subset_sc = ss.fit_transform(X_noisy_subset)

            # Data Loader Generation
            noisy_subset_dataloader = get_data_loader(X_noisy_subset_sc, y_subset, shuffle=True)

            train_epoch_loss, train_epoch_acc, val_epoch_acc, val_epoch_loss, _, model = train(model=model,
                                                                                               train_loader=noisy_subset_dataloader,
                                                                                               val_loader=self.valid_noisy_dl,
                                                                                               test_loader=self.test_noisy_dl,
                                                                                               optimizer=optimizer,
                                                                                               criterion=criterion,
                                                                                               device=device,
                                                                                               flatten=True,
                                                                                               verbose=False,
                                                                                               early_stopping=False,
                                                                                               epochs=10)

            # Evaluation
            test_epoch_acc = evaluate(model, noisy_test_dataloader, device, flatten=True)

            print("Boosting epoch: {} | Train accuracy: {} | train loss: {} | test accuracy: {} ".format(epoch,
                                                                                                         train_epoch_acc[
                                                                                                             -1],
                                                                                                         train_epoch_loss[
                                                                                                             -1],
                                                                                                         test_epoch_acc))

            for tr_l in train_epoch_loss:
                train_losses.append(tr_l)
            for tr_a in train_epoch_acc:
                train_accuracies.append(tr_a)
            for v_a in val_epoch_acc:
                val_accuracies.append(v_a)

            weights_boosting = update_weights_boosting(model=model, weights_boosting=weights_boosting,
                                                       X=X_noisy_train_scaled,
                                                       y=self.y_train, device=device, lr=lr, mode=mode)
            criterion.weights_boosting = weights_boosting

            print('Sanity check | Post-training evaluation of a single sample...')
            self.evaluate_single_sample(epoch, model, X_noisy_train_scaled, self.y_train, idx=rnd_idx,
                                        weights_boosting=weights_boosting,
                                        lr=lr, loss=criterion)

            exit_to_monitor, w_to_monitor = self.evaluate_single_sample(epoch, model, X_noisy_train_scaled,
                                                                        self.y_train,
                                                                        idx=to_monitor,
                                                                        weights_boosting=weights_boosting,
                                                                        lr=lr, loss=criterion)

            to_monitor_weights.append(w_to_monitor)
            to_monitor_predictions.append(exit_to_monitor)

        return train_accuracies, train_losses, val_accuracies, weights_boosting, to_monitor_weights, to_monitor_predictions

    def run_simple_boosting(self, epochs, perc=None, threshold=None):
        if not perc:
            perc = 1.0
        # Convert to [-1, 1]
        y = convert_labels(self.y, old_labels=[0, 1], new_labels=[-1, 1])

        # Training, Validation and Test split
        (X_train, y_train), (X_valid, y_valid) = dataset_split(self.X, self.y, return_data='samples')

        # Scaling of X
        ss = StandardScaler()
        X_train_scaled = ss.fit_transform(X_train)
        X_valid_scaled = ss.transform(X_valid)

        # TODO Mapping X to y: better come up with a smarter solution
        X_to_y = dict(zip([tuple(x.flatten()) for x in X_train_scaled], y_train))

        train_dataloader = get_data_loader(X_train_scaled, y_train)
        val_dataloader = get_data_loader(X_valid_scaled, y_valid)

        torch.manual_seed(RANDOM_SEED)
        model = FFSimpleNet(input_dim=self.n_features, output_dim=1)
        criterion, weights_boosting = init_exponential_loss(X_train_scaled)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        train_accuracies = []
        train_losses = []
        val_accuracies = []

        epsilon_collection = []
        learning_rate_collection = []
        weights = []
        weights.append(copy.deepcopy(weights_boosting))

        training_set_sizes = []

        for epoch in range(epochs):
            # TODO
            points_weights = list(weights_boosting.items())
            if not threshold:
                points_weights = sorted(points_weights, key=lambda x: x[1])[::-1][:int(perc * len(points_weights))]
            else:
                points_weights = [pw for pw in points_weights if pw[1] > threshold]
            points = [pw[0] for pw in points_weights]

            training_set_size = len(points)
            training_set_sizes.append(training_set_size)

            y_points = np.array([X_to_y[p] for p in points])
            X_points = np.array([np.array(p) for p in points])
            train_dataloader_subset = get_data_loader(X_points, y_points)

            for i in range(10):
                train_loss, train_acc = train(model, train_dataloader_subset, optimizer, criterion, device)

            weights_boosting, epsilon, learning_rate, epsilon = update_weights_boosting(model,
                                                                                        weights_boosting,
                                                                                        X_train_scaled, y_train,
                                                                                        learning_rate_collection,
                                                                                        epsilon_collection, device)
            criterion.weights_boosting = weights_boosting
            val_acc = evaluate(model, val_dataloader, device)

            weights.append(copy.deepcopy(weights_boosting))
            train_accuracies.append(train_acc)
            train_losses.append(train_loss)
            val_accuracies.append(val_acc)

            print('Epoch {} | Train acc: {:.3f} | Train loss: {:.3f} | Valid acc: {:.3f}'.format(epoch,
                                                                                                 train_acc,
                                                                                                 train_loss,
                                                                                                 val_acc))

        # generate_simple_boosting_plots(epochs, train_accuracies, train_losses, val_accuracies, weights,
        #                                learning_rate_collection,
        #                                epsilon_collection)

        return train_accuracies, train_losses, val_accuracies, weights, learning_rate_collection, epsilon_collection, training_set_sizes

    def toy_run_recompute(self, model, X_noisy_train, y_train, X_noisy_test, y_test, total_epochs, prop,
                          property_perc, most, random=False, alternate=False):
        decision_boundaries = []
        train_losses = []
        train_accuracies = []
        test_accuracies = []
        subsets = []

        # Scaling the test set
        ss = StandardScaler()
        ss.fit(X_noisy_train)
        X_noisy_test_scaled = ss.transform(X_noisy_test)
        noisy_test_dataloader = get_data_loader(X_noisy_test_scaled, y_test, shuffle=False)

        # Getting the starting point of the decision boundary
        params = model.state_dict()
        weight1 = params['f1.weight'][0][0].numpy()
        weight2 = params['f1.weight'][0][1].numpy()
        bias = params['f1.bias'][0].numpy()

        initial_DB = [weight1, weight2, bias]
        decision_boundaries.append(initial_DB)

        new_model = copy.deepcopy(model)
        criterion = WeightedExponentialLoss()
        optimizer = torch.optim.SGD(new_model.parameters(), lr=0.01)

        if random:
            X_noisy_subset, y_subset = get_random_subset(X_noisy_train, y_train, property_perc)

        for epoch in range(total_epochs):
            if alternate:
                if epoch % 2 == 0:
                    X_noisy_subset, y_subset = get_samples_by_property(new_model, X_noisy_train, y_train,
                                                                       property_perc, most=most,
                                                                       prop=prop)
                else:
                    X_noisy_subset, y_subset = get_random_subset(X_noisy_train, y_train, property_perc)
            else:
                if not random:
                    X_noisy_subset, y_subset = get_samples_by_property(new_model, X_noisy_train, y_train,
                                                                       property_perc, most=most,
                                                                       prop=prop)
            # Rescale the input
            X_noisy_subset_scaled = StandardScaler().fit_transform(X_noisy_subset)
            subsets.append([X_noisy_subset_scaled, y_subset])

            # Data Loader Generation with weighted sampler
            dataset = ToyDataset(X=X_noisy_subset_scaled, y=y_subset)
            weighted_sampler = WeightedSampler(dataset=dataset)
            weighted_sampler.update_weights(model=model, dataset=dataset, learning_rate=1e-3)
            noisy_subset_dataloader = get_data_loader(X_noisy_subset_scaled, y_subset, sampler=weighted_sampler,
                                                      shuffle=False)

            # One epoch train
            train_epoch_loss, train_epoch_acc = train(new_model, noisy_subset_dataloader, optimizer, criterion, device)

            # Evaluation
            test_epoch_acc = evaluate(new_model, noisy_test_dataloader, device)

            # Getting the new decision boundary
            new_params = new_model.state_dict()
            new_weight1 = new_params['f1.weight'][0][0].numpy().copy()
            new_weight2 = new_params['f1.weight'][0][1].numpy().copy()
            new_bias = new_params['f1.bias'][0].numpy().copy()

            new_DB = [new_weight1, new_weight2, new_bias]

            decision_boundaries.append(new_DB)
            train_losses.append(train_epoch_loss)
            train_accuracies.append(train_epoch_acc)
            test_accuracies.append(test_epoch_acc)

        return decision_boundaries, train_losses, train_accuracies, test_accuracies, subsets

    def visualize_dbs(self, n_epochs, prop, property_perc, alternate, animation_name='anim'):

        X_train_noisy_sc = StandardScaler().fit_transform(self.X_train_noisy)

        print('Creating animations...', end='')
        animator = Animator(X_train_noisy_sc, self.y_train, prop, property_perc, alternate, n_epochs, interval=100)

        criterion, weights_boosting = init_exponential_loss(X_train_noisy_sc)

        dbs_rand, train_losses_rand, train_accs_rand, \
        test_accs_rand, subsets_rand = self.toy_run_recompute(
            self.model_clean, self.X_train_noisy, self.y_train, self.X_test_noisy, self.y_test, n_epochs,
            criterion=criterion, prop=prop, property_perc=property_perc,
            most=True, random=True, alternate=False)
        dbs_top, train_losses_top, train_accs_top, \
        test_accs_top, subsets_top = self.toy_run_recompute(
            self.model_clean, self.X_train_noisy, self.y_train, self.X_test_noisy, self.y_test, n_epochs,
            criterion=criterion, prop=prop, property_perc=property_perc,
            most=True, random=False, alternate=alternate)
        dbs_bottom, train_losses_bottom, train_accs_bottom, \
        test_accs_bottom, subsets_bottom = self.toy_run_recompute(
            self.model_clean, self.X_train_noisy, self.y_train, self.X_test_noisy, self.y_test, n_epochs,
            criterion=criterion, prop=prop, property_perc=property_perc,
            most=False, random=False, alternate=alternate)
        dbs_all, train_losses_all, train_accs_all, \
        test_accs_all, subsets_all = self.toy_run_recompute(
            self.model_clean,
            self.X_train_noisy, self.y_train, self.X_test_noisy, self.y_test, n_epochs,
            criterion=criterion, prop=prop, property_perc=1.0,
            most=False, random=True, alternate=False)

        dbs = [dbs_rand, dbs_top, dbs_bottom, dbs_all]
        train_accs = [train_accs_rand, train_accs_top, train_accs_bottom, train_accs_all]
        test_accs = [test_accs_rand, test_accs_top, test_accs_bottom, test_accs_all]

        self.y_train = convert_labels(self.y_train, [-1, 1], [0, 1])
        self.y_valid = convert_labels(self.y_valid, [-1, 1], [0, 1])
        self.y_test = convert_labels(self.y_test, [-1, 1], [0, 1])
        criterion = nn.BCELoss()
        self.model_clean = FFSimpleNet(input_dim=self.n_features, output_dim=1, activation='sigmoid')
        optimizer = torch.optim.SGD(self.model_clean.parameters(), lr=0.01)
        self.train_accs_clean, self.val_accs_clean, self.test_accs_clean, epoch, model = early_stopping(
            self.model_clean, self.train_dl, self.valid_dl, self.test_dl, criterion=criterion,
            optimizer=optimizer, device=device, verbose=False)

        dbs_rand, train_losses_rand, train_accs_rand, \
        test_accs_rand, subsets_rand = self.toy_run_recompute(
            self.model_clean, self.X_train_noisy, self.y_train, self.X_test_noisy, self.y_test, n_epochs,
            criterion=criterion, prop=prop, property_perc=property_perc,
            most=True, random=True, alternate=False)
        dbs_top, train_losses_top, train_accs_top, \
        test_accs_top, subsets_top = self.toy_run_recompute(
            self.model_clean, self.X_train_noisy, self.y_train, self.X_test_noisy, self.y_test, n_epochs,
            criterion=criterion, prop=prop, property_perc=property_perc,
            most=True, random=False, alternate=alternate)
        dbs_bottom, train_losses_bottom, train_accs_bottom, \
        test_accs_bottom, subsets_bottom = self.toy_run_recompute(
            self.model_clean, self.X_train_noisy, self.y_train, self.X_test_noisy, self.y_test, n_epochs,
            criterion=criterion, prop=prop, property_perc=property_perc,
            most=False, random=False, alternate=alternate)
        dbs_all, train_losses_all, train_accs_all, \
        test_accs_all, subsets_all = self.toy_run_recompute(
            self.model_clean,
            self.X_train_noisy, self.y_train, self.X_test_noisy, self.y_test, n_epochs,
            criterion=criterion, prop=prop, property_perc=1.0,
            most=False, random=True, alternate=False)

        dbs = dbs + [dbs_rand, dbs_top, dbs_bottom, dbs_all]
        train_accs = train_accs + [train_accs_rand, train_accs_top, train_accs_bottom, train_accs_all]
        test_accs = test_accs + [test_accs_rand, test_accs_top, test_accs_bottom, test_accs_all]
        labels = ['rand_exp', 'top_exp', 'bottom_exp', 'all_exp', 'rand_bce', 'top_bce', 'bottom_bce', 'all_bce']
        colors = ['orange', 'green', 'red', 'blue', 'orange', 'green', 'red', 'blue']
        markers = ['--', '--', '--', '--', '.-', '.-', '.-', '.-']
        animation = animator.run(dbs, train_accs, test_accs, labels, colors, markers)

        print('done!')
        print('Saving animation...', end='')
        animation.save('animations/{}={}_{}={}.gif'.format(prop, property_perc, 'alternate', alternate), dpi=100)
        print('done!')

    def evaluate_single_sample(self, epoch, model, X_train, y_train, idx, weights_boosting, lr, loss):
        model.eval()
        x, y_true = X_train[idx], y_train[idx]
        x = copy.deepcopy(x)
        x = np.moveaxis(x, source=-1, destination=0).astype(np.float32)
        x = np.expand_dims(x, axis=0)
        x = torch.from_numpy(x)
        x = x.to(device)
        model = model.to(device)

        y_hat = model(x.float()).cpu()

        if loss == 'exp':
            y_hat = torch.where(y_hat > 0, torch.ones(1), torch.ones(1) * (-1))
        else:
            if y_hat.shape[1] <= 1:
                y_hat = torch.where(y_hat > 0.5, torch.ones(1), torch.zeros(1))
            else:
                y_hat = torch.argmax(y_hat.data, 1)

        w = weights_boosting[tuple(X_train[idx].flatten())]

        if y_hat == y_true:
            exit = 1
        else:
            exit = 0

        print('Epoch: {} | Sample index: {} | Y_hat: {} | Y_true: {} | Weight: {} | Lr: {}'.format(epoch, idx, y_hat,
                                                                                                   y_true,
                                                                                                   w, lr))
        return exit, w

