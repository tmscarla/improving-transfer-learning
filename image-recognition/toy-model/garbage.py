def run(self):
    X, y = make_classification(n_samples=self.n_samples, n_features=self.n_features, n_informative=2, n_redundant=0,
                               n_classes=self.n_classes, n_clusters_per_class=1)

    plt.cool()
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=40, edgecolor='k')

    # Training, Validation and Test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_PERCENTAGE, random_state=RANDOM_SEED)
    training_idxs, test_idxs = next(sss.split(X, y))

    X_train, X_test = X[training_idxs], X[test_idxs]
    y_train, y_test = y[training_idxs], y[test_idxs]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_PERCENTAGE, random_state=RANDOM_SEED)
    training_idxs, validation_idxs = next(sss.split(X_train, y_train))

    X_train, X_valid = X_train[training_idxs], X_train[validation_idxs]
    y_train, y_valid = y_train[training_idxs], y_train[validation_idxs]

    # Scaling of X
    ss = StandardScaler()
    X_train_scaled = ss.fit_transform(X_train)
    X_valid_scaled = ss.transform(X_valid)
    X_test_scaled = ss.transform(X_test)

    train_dataloader = get_data_loader(X_train_scaled, y_train)
    val_dataloader = get_data_loader(X_valid_scaled, y_valid)
    test_dataloader = get_data_loader(X_test_scaled, y_test)

    print('Training clean model...')
    model = FFSimpleNet(input_dim=self.n_features, output_dim=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    train_accuracies, val_accuracies, test_accuracies, epoch, model = early_stopping(model, train_dataloader,
                                                                                     val_dataloader,
                                                                                     test_dataloader, optimizer,
                                                                                     device)

    plt.figure(figsize=(8, 8))
    plt.plot(range(epoch), train_accuracies, label='Train')
    plt.plot(range(epoch), val_accuracies, label='Validation')
    plt.plot(range(epoch), test_accuracies, label='Test')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='upper left')

    X_s = StandardScaler().fit_transform(X)
    X_noisy = rotate(X_s, theta=120.0)

    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.title('Original scaled data scatter plot')
    plt.scatter(X_s[:, 0], X_s[:, 1], marker='o', c=y,
                s=40, edgecolor='k')

    plt.subplot(1, 2, 2)
    plt.title('Noisy data scatter plot')
    plt.scatter(X_noisy[:, 0], X_noisy[:, 1], marker='o', c=y,
                s=40, edgecolor='k')

    # Training, Validation and Test split for noisy samples
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_PERCENTAGE, random_state=RANDOM_SEED)
    training_idxs, test_idxs = next(sss.split(X_noisy, y))

    X_noisy_train, X_noisy_test = X_noisy[training_idxs], X_noisy[test_idxs]
    y_train, y_test = y[training_idxs], y[test_idxs]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_PERCENTAGE, random_state=RANDOM_SEED)
    training_idxs, validation_idxs = next(sss.split(X_noisy_train, y_train))

    X_noisy_train, X_noisy_valid = X_noisy_train[training_idxs], X_noisy_train[validation_idxs]
    y_train, y_valid = y_train[training_idxs], y_train[validation_idxs]

    # Selection of top, random, bottom samples per property
    X_noisy_top, y_top = get_samples_by_property(model, X_noisy_train, y_train,
                                                 self.property_perc, most=True,
                                                 prop=self.property)
    X_noisy_random, y_random = get_random_subset(X_noisy_train, y_train, self.property_perc)
    X_noisy_bottom, y_bottom = get_samples_by_property(model, X_noisy_train, y_train,
                                                       self.property_perc, most=False,
                                                       prop=self.property)

    X_noisy_viz = StandardScaler().fit_transform(X_noisy)

    ss = StandardScaler()
    ss.fit(X_noisy_train)

    X_noisy_top_viz = ss.transform(X_noisy_top)
    X_noisy_random_viz = ss.transform(X_noisy_random)
    X_noisy_bottom_viz = ss.transform(X_noisy_bottom)

    weight1, weight2, bias = get_params(model)

    plt.figure(figsize=(20, 4))
    plt.subplot(1, 5, 1)
    plt.title('Original scaled data scatter plot')
    plt.scatter(X_s[:, 0], X_s[:, 1], marker='o', c=y,
                s=40, edgecolor='k')
    abline(weight1, weight2, bias, label='Original DB')
    plt.legend(loc='best')

    plt.subplot(1, 5, 2)
    plt.title('Noisy scaled data scatter plot')
    plt.scatter(X_noisy_viz[:, 0], X_noisy_viz[:, 1], marker='o', c=y,
                s=40, edgecolor='k')
    abline(weight1, weight2, bias, label='Original DB')
    plt.legend(loc='best')

    plt.subplot(1, 5, 3)
    plt.title('Top {} scaled - {}'.format(self.property_perc, self.property))
    plt.scatter(X_noisy_top_viz[:, 0], X_noisy_top_viz[:, 1], marker='o', c=y_top,
                s=40, edgecolor='k')
    abline(weight1, weight2, bias, label='Original DB')
    plt.legend(loc='best')

    plt.subplot(1, 5, 4)
    plt.title('Random {} scaled - {}'.format(self.property_perc, self.property))
    plt.scatter(X_noisy_random_viz[:, 0], X_noisy_random_viz[:, 1], marker='o', c=y_random,
                s=40, edgecolor='k')
    abline(weight1, weight2, bias, label='Original DB')
    plt.legend(loc='best')

    plt.subplot(1, 5, 5)
    plt.title('Bottom {} scaled - {}'.format(self.property_perc, self.property))
    plt.scatter(X_noisy_bottom_viz[:, 0], X_noisy_bottom_viz[:, 1], marker='o', c=y_bottom,
                s=40, edgecolor='k')
    abline(weight1, weight2, bias, label='Original DB')
    plt.legend(loc='best')

    # RESCALE TO TRAIN
    X_noisy_top_s = StandardScaler().fit_transform(X_noisy_top)
    X_noisy_random_s = StandardScaler().fit_transform(X_noisy_random)
    X_noisy_bottom_s = StandardScaler().fit_transform(X_noisy_bottom)
    X_noisy_train_s = StandardScaler().fit_transform(X_noisy_train)
    X_noisy_s = StandardScaler().fit_transform(X_noisy)
    X_s = StandardScaler().fit_transform(X)

    noisy_top_dataloader = get_data_loader(X_noisy_top_s, y_top)
    noisy_random_dataloader = get_data_loader(X_noisy_random_s, y_random)
    noisy_bottom_dataloader = get_data_loader(X_noisy_bottom_s, y_bottom)

    # Top Train and Validation split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_PERCENTAGE, random_state=RANDOM_SEED)
    training_idxs, validation_idxs = next(sss.split(X_noisy_top, y_top))

    X_noisy_top_train, X_noisy_top_valid = X_noisy_top[training_idxs], X_noisy_top[validation_idxs]
    y_top_train, y_top_valid = y_top[training_idxs], y_top[validation_idxs]

    # Random Train and Validation split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_PERCENTAGE, random_state=RANDOM_SEED)
    training_idxs, validation_idxs = next(sss.split(X_noisy_random, y_random))

    X_noisy_random_train, X_noisy_random_valid = X_noisy_random[training_idxs], X_noisy_random[validation_idxs]
    y_random_train, y_random_valid = y_random[training_idxs], y_random[validation_idxs]

    # Bottom Train and Validation split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_PERCENTAGE, random_state=RANDOM_SEED)
    training_idxs, validation_idxs = next(sss.split(X_noisy_bottom, y_bottom))

    X_noisy_bottom_train, X_noisy_bottom_valid = X_noisy_bottom[training_idxs], X_noisy_bottom[validation_idxs]
    y_bottom_train, y_bottom_valid = y_bottom[training_idxs], y_bottom[validation_idxs]

    # Scaling of X_noisy
    ss_top = StandardScaler()
    X_noisy_top_train_scaled = ss_top.fit_transform(X_noisy_top_train)
    X_noisy_top_valid_scaled = ss_top.transform(X_noisy_top_valid)

    ss_random = StandardScaler()
    X_noisy_random_train_scaled = ss_random.fit_transform(X_noisy_random_train)
    X_noisy_random_valid_scaled = ss_random.transform(X_noisy_random_valid)

    ss_bottom = StandardScaler()
    X_noisy_bottom_train_scaled = ss_bottom.fit_transform(X_noisy_bottom_train)
    X_noisy_bottom_valid_scaled = ss_bottom.transform(X_noisy_bottom_valid)

    X_noisy_test_top_scaled = ss_top.transform(X_noisy_test)
    X_noisy_test_random_scaled = ss_random.transform(X_noisy_test)
    X_noisy_test_bottom_scaled = ss_bottom.transform(X_noisy_test)

    train_noisy_top_dataloader = get_data_loader(X_noisy_top_train_scaled, y_top_train)
    valid_noisy_top_dataloader = get_data_loader(X_noisy_top_valid_scaled, y_top_valid)

    train_noisy_random_dataloader = get_data_loader(X_noisy_random_train_scaled, y_random_train)
    valid_noisy_random_dataloader = get_data_loader(X_noisy_random_valid_scaled, y_random_valid)

    train_noisy_bottom_dataloader = get_data_loader(X_noisy_bottom_train_scaled, y_bottom_train)
    valid_noisy_bottom_dataloader = get_data_loader(X_noisy_bottom_valid_scaled, y_bottom_valid)

    test_noisy_top_dataloader = get_data_loader(X_noisy_test_top_scaled, y_test)
    test_noisy_random_dataloader = get_data_loader(X_noisy_test_random_scaled, y_test)
    test_noisy_bottom_dataloader = get_data_loader(X_noisy_test_bottom_scaled, y_test)

    print('Finetuning on top samples...')
    model_top = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model_top.parameters(), lr=0.01)

    train_top_accuracies, val_top_accuracies, test_top_accuracies, epoch_top, model_top = early_stopping(model_top,
                                                                                                         train_noisy_top_dataloader,
                                                                                                         valid_noisy_top_dataloader,
                                                                                                         test_noisy_top_dataloader,
                                                                                                         optimizer,
                                                                                                         device)
    print('Finetuning on random samples...')
    # Finetuning the model on random samples
    model_random = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model_random.parameters(), lr=0.01)

    train_random_accuracies, val_random_accuracies, test_random_accuracies, epoch_random, model_random = early_stopping(
        model_random,
        train_noisy_random_dataloader,
        valid_noisy_random_dataloader,
        test_noisy_random_dataloader,
        optimizer,
        device)

    print('Finetuning on bottom samples...')
    # Finetuning the model on bottom samples
    model_bottom = copy.deepcopy(model)
    optimizer = torch.optim.SGD(model_bottom.parameters(), lr=0.01)

    train_bottom_accuracies, val_bottom_accuracies, test_bottom_accuracies, epoch_bottom, model_bottom = early_stopping(
        model_bottom,
        train_noisy_bottom_dataloader,
        valid_noisy_bottom_dataloader,
        test_noisy_bottom_dataloader,
        optimizer,
        device)

    weight1_top, weight2_top, bias_top = get_params(model_top)
    weight1_random, weight2_random, bias_random = get_params(model_random)
    weight1_bottom, weight2_bottom, bias_bottom = get_params(model_bottom)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.title('Original data scatter plot')
    plt.scatter(X[:, 0], X[:, 1], marker='o', c=y,
                s=40, edgecolor='k', alpha=0.3)
    abline(weight1, weight2, bias, label='Original DB')
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.title('Original rescaled data scatter plot')
    plt.scatter(X_s[:, 0], X_s[:, 1], marker='o', c=y,
                s=40, edgecolor='k', alpha=0.3)
    abline(weight1, weight2, bias, label='Original DB')
    plt.legend(loc='best')

    # Training, Validation and Test split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_PERCENTAGE, random_state=RANDOM_SEED)
    training_idxs, test_idxs = next(sss.split(X_noisy, y))

    X_train_best, X_test_best = X_noisy[training_idxs], X_noisy[test_idxs]
    y_train_best, y_test_best = y[training_idxs], y[test_idxs]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_PERCENTAGE, random_state=RANDOM_SEED)
    training_idxs, validation_idxs = next(sss.split(X_train_best, y_train_best))

    X_train_best, X_valid_best = X_train_best[training_idxs], X_train_best[validation_idxs]
    y_train_best, y_valid_best = y_train_best[training_idxs], y_train_best[validation_idxs]

    ss = StandardScaler()
    # Scaling of X
    X_train_best_scaled = ss.fit_transform(X_train_best)
    X_valid_best_scaled = ss.transform(X_valid_best)
    X_test_best_scaled = ss.transform(X_test_best)

    train_best_dataloader = get_data_loader(X_train_best_scaled, y_train_best)
    val_best_dataloader = get_data_loader(X_valid_best_scaled, y_valid_best)
    test_best_dataloader = get_data_loader(X_test_best_scaled, y_test_best)

    model_best = FFSimpleNet(input_dim=self.n_features, output_dim=1)
    optimizer = torch.optim.SGD(model_best.parameters(), lr=0.01)
    train_accuracies, val_accuracies, test_accuracies, epoch, model_best = early_stopping(model_best,
                                                                                          train_best_dataloader,
                                                                                          val_best_dataloader,
                                                                                          test_best_dataloader,
                                                                                          optimizer, device)

    weight1_noisy, weight2_noisy, bias_noisy = get_params(model_best)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Original rescaled data scatter plot')
    plt.scatter(X_s[:, 0], X_s[:, 1], marker='o', c=y,
                s=40, edgecolor='k', alpha=0.3)
    abline(weight1, weight2, bias, label='Original DB')
    plt.legend(loc='best')
    plt.axis([-6, 6, -4, 4])

    plt.subplot(1, 3, 2)
    plt.title('Noisy rescaled data scatter plot')
    plt.scatter(X_noisy_s[:, 0], X_noisy_s[:, 1], marker='o', c=y,
                s=40, edgecolor='k', alpha=0.3)
    abline(weight1, weight2, bias, label='Original DB')
    abline(weight1_noisy, weight2_noisy, bias_noisy, label='Best DB')
    plt.legend(loc='best')
    plt.axis([-6, 6, -4, 4])

    plt.subplot(1, 3, 3)
    plt.title('Noisy training rescaled data scatter plot')
    plt.scatter(X_noisy_train_s[:, 0], X_noisy_train_s[:, 1], marker='o', c=y_train,
                s=40, edgecolor='k', alpha=0.3)
    abline(weight1, weight2, bias, label='Original DB')
    abline(weight1_noisy, weight2_noisy, bias_noisy, label='Best DB')
    plt.legend(loc='best')
    plt.axis([-6, 6, -4, 4])

    plt.figure(figsize=(15, 10))
    plt.subplot(2, 3, 1)
    plt.title('Top {} plot - {}'.format(self.property_perc, self.property))
    plt.scatter(X_noisy_top_viz[:, 0], X_noisy_top_viz[:, 1], marker='o', c=y_top,
                s=40, edgecolor='k', alpha=0.3)
    abline(weight1, weight2, bias, label='Original DB')
    abline(weight1_top, weight2_top, bias_top, label='Top DB')
    plt.legend(loc='best')
    plt.axis([-6, 6, -4, 4])

    plt.subplot(2, 3, 2)
    plt.title('Random {} plot - {}'.format(self.property_perc, self.property))
    plt.scatter(X_noisy_random_viz[:, 0], X_noisy_random_viz[:, 1], marker='o', c=y_random,
                s=40, edgecolor='k', alpha=0.3)
    abline(weight1, weight2, bias, label='Original DB')
    abline(weight1_random, weight2_random, bias_random, label='Random DB')
    plt.legend(loc='best')
    plt.axis([-6, 6, -4, 4])

    plt.subplot(2, 3, 3)
    plt.title('Bottom {} plot - {}'.format(self.property_perc, self.property))
    plt.scatter(X_noisy_bottom_viz[:, 0], X_noisy_bottom_viz[:, 1], marker='o', c=y_bottom, alpha=0.3,
                s=40, edgecolor='k')
    abline(weight1, weight2, bias, label='Original DB')
    abline(weight1_bottom, weight2_bottom, bias_bottom, label='Bottom DB')
    plt.legend(loc='best')
    plt.axis([-6, 6, -4, 4])

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title('Noisy scaled data scatter plot - Top DB')
    plt.scatter(X_noisy_s[:, 0], X_noisy_s[:, 1], marker='o', c=y,
                s=40, edgecolor='k')
    abline(weight1, weight2, bias, label='Original DB')
    abline(weight1_top, weight2_top, bias_top, label='Top DB')
    plt.legend(loc='best')
    plt.axis([-6, 6, -6, 6])

    plt.subplot(1, 3, 2)
    plt.title('Noisy scaled data scatter plot - Random DB')
    plt.scatter(X_noisy_s[:, 0], X_noisy_s[:, 1], marker='o', c=y,
                s=40, edgecolor='k')
    abline(weight1, weight2, bias, label='Original DB')
    abline(weight1_random, weight2_random, bias_random, label='Random DB')
    plt.legend(loc='best')
    plt.axis([-6, 6, -6, 6])

    plt.subplot(1, 3, 3)
    plt.title('Noisy scaled data scatter plot - Bottom DB')
    plt.scatter(X_noisy_s[:, 0], X_noisy_s[:, 1], marker='o', c=y,
                s=40, edgecolor='k')
    abline(weight1, weight2, bias, label='Original DB')
    abline(weight1_bottom, weight2_bottom, bias_bottom, label='Bottom DB')
    plt.legend(loc='best')
    plt.axis([-6, 6, -6, 6])

    plt.figure(figsize=(18, 5))

    plt.subplot(1, 3, 1)
    plt.plot(range(epoch_top), train_top_accuracies, label='Train')
    plt.plot(range(epoch_top), val_top_accuracies, label='Validation')
    plt.plot(range(epoch_top), test_top_accuracies, label='Test')
    plt.title('Top {} - {}'.format(self.property_perc, self.property), fontsize=16)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='upper left')

    plt.subplot(1, 3, 2)
    plt.plot(range(epoch_random), train_random_accuracies, label='Train')
    plt.plot(range(epoch_random), val_random_accuracies, label='Validation')
    plt.plot(range(epoch_random), test_random_accuracies, label='Test')
    plt.title('Random {} - {}'.format(self.property_perc, self.property), fontsize=16)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='upper left')

    plt.subplot(1, 3, 3)
    plt.plot(range(epoch_bottom), train_bottom_accuracies, label='Train')
    plt.plot(range(epoch_bottom), val_bottom_accuracies, label='Validation')
    plt.plot(range(epoch_bottom), test_bottom_accuracies, label='Test')
    plt.title('Bottom {} - {}'.format(self.property_perc, self.property), fontsize=16)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='upper left')

    plt.show()

    ######################## ANIMATIONS ########################
    from animator import Animator

    property_perc = 0.1
    n_epochs = 100
    prop = 'entropy'
    alternate = True

    print('Creating animations...')
    animator = Animator(X_noisy_train_s, y_train, prop, property_perc, alternate, n_epochs, interval=100)

    decision_boundaries_rand, train_losses_rand, train_accuracies_rand, test_accuracies_rand, subsets_rand = self.toy_run_recompute(
        model, X_noisy_train, y_train,
        X_noisy_test, y_test, n_epochs,
        prop=prop, property_perc=property_perc,
        most=True, random=True, alternate=False)
    decision_boundaries_top, train_losses_top, train_accuracies_top, test_accuracies_top, subsets_top = self.toy_run_recompute(
        model, X_noisy_train, y_train,
        X_noisy_test, y_test, n_epochs,
        prop=prop, property_perc=property_perc,
        most=True, random=False, alternate=alternate)
    decision_boundaries_bottom, train_losses_bottom, train_accuracies_bottom, test_accuracies_bottom, subsets_bottom = self.toy_run_recompute(
        model, X_noisy_train, y_train,
        X_noisy_test, y_test, n_epochs,
        prop=prop, property_perc=property_perc,
        most=False, random=False, alternate=alternate)
    decision_boundaries_all, train_losses_all, train_accuracies_all, test_accuracies_all, subsets_all = self.toy_run_recompute(
        FFSimpleNet(input_dim=self.n_features, output_dim=1),
        X_noisy_train, y_train,
        X_noisy_test, y_test, n_epochs,
        prop=prop, property_perc=1.0,
        most=False, random=True, alternate=False)

    animation = animator.run(decision_boundaries_rand, train_accuracies_rand, test_accuracies_rand,
                             decision_boundaries_top, train_accuracies_top, test_accuracies_top,
                             decision_boundaries_bottom, train_accuracies_bottom, test_accuracies_bottom,
                             decision_boundaries_all, train_accuracies_all, test_accuracies_all)
    animation.save('animations/{}={}_{}={}.gif'.format(prop, property_perc, 'alternate', alternate), dpi=100)

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