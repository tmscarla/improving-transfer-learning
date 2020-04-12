from toy_experiment import *
import warnings
warnings.filterwarnings("ignore")
import datetime


if __name__ == '__main__':
    N_EPOCHS = 10
    property_perc = 0.20
    theta = 120.0
    N_SAMPLES = 1000
    top_test, rnd_test, bottom_test = [], [], []

    n_runs = 5
    for i in range(n_runs):
        print('####################### RUN: {}'.format(i))
        toy = ToyExperiment(n_features=2, n_classes=2, n_samples=N_SAMPLES, N_EPOCHS=N_EPOCHS, alternate=False, property='entropy',
                            property_perc=property_perc, theta=theta, most=True)

        top_full_train_accuracies, top_train_accuracies, top_train_losses, top_test_accuracies, top_ensemble_test_accuracies, top_lr, top_w_acc = toy.toy_run_boosting()
        top_test.append(top_test_accuracies)
        print('\n----------------------------------------------------------------------------\n')
        rnd_full_train_accuracies, rnd_train_accuracies, rnd_train_losses, rnd_test_accuracies, rnd_ensemble_test_accuracies, rnd_lr, rand_w_acc = toy.toy_run_boosting(random=True)
        rnd_test.append(rnd_test_accuracies)
        print('\n----------------------------------------------------------------------------\n')
        toy.most = False
        bottom_full_train_accuracies, bottom_train_accuracies, bottom_train_losses, bottom_test_accuracies, bottom_ensemble_test_accuracies, bottom_lr, bottom_w_acc =toy.toy_run_boosting()
        bottom_test.append(bottom_test_accuracies)

    datetime = datetime.datetime.now().strftime("%H-%M-%S")

    plt.figure()
    top_mean = np.mean(top_test, axis=0)
    top_std = np.std(top_test, axis=0)
    plt.plot(range(N_EPOCHS), top_mean, label='Top', alpha=0.9)
    plt.fill_between(range(N_EPOCHS), top_mean-top_std, top_mean+top_std, alpha=0.15)

    rnd_mean = np.mean(rnd_test, axis=0)
    rnd_std = np.std(rnd_test, axis=0)
    plt.plot(range(N_EPOCHS), rnd_mean, label='Random', alpha=0.9)
    plt.fill_between(range(N_EPOCHS), rnd_mean-rnd_std, rnd_mean+rnd_std, alpha=0.15)

    bottom_mean = np.mean(bottom_test, axis=0)
    bottom_std = np.std(bottom_test, axis=0)
    plt.plot(range(N_EPOCHS), bottom_mean, label='Bottom', alpha=0.9)
    plt.fill_between(range(N_EPOCHS), bottom_mean-bottom_std, bottom_mean+bottom_std, alpha=0.15)

    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Test accuracy of each model of the ensemble over {} runs | theta = 120'.format(n_runs))
    plt.legend(loc='upper left')
    plt.savefig('mean_std_120.png')
    exit()

    plt.figure(figsize=(20, 7))

    plt.subplot(1,3,1)
    plt.plot(range(N_EPOCHS), top_train_accuracies, label='Top')
    plt.plot(range(N_EPOCHS), rnd_train_accuracies, label='Random')
    plt.plot(range(N_EPOCHS), bottom_train_accuracies, label='Bottom')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='upper left')
    plt.title('Train accuracy of each model of the ensemble\n'
              'on its subset'
              '\n size of dataset: {} \n selected entropy = {}% | theta = {}'.format(N_SAMPLES, property_perc * 100, theta))

    plt.subplot(1, 3, 2)
    plt.plot(range(N_EPOCHS), top_train_losses, label='Top')
    plt.plot(range(N_EPOCHS), rnd_train_losses, label='Random')
    plt.plot(range(N_EPOCHS), bottom_train_losses, label='Bottom')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Train loss of each model of the ensemble on whole training set'
              '\n size of dataset: {} \n selected entropy = {}% | theta = {}'.format(N_SAMPLES, property_perc * 100,
                                                                                     theta))
    plt.legend(loc='upper left')

    plt.subplot(1, 3, 3)
    plt.plot(range(N_EPOCHS), top_full_train_accuracies, label='Top')
    plt.plot(range(N_EPOCHS), rnd_full_train_accuracies, label='Random')
    plt.plot(range(N_EPOCHS), bottom_full_train_accuracies, label='Bottom')
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Accuracy of each model of the ensemble on whole training set'
              '\n size of dataset: {} \n selected entropy = {}% | theta = {}'.format(N_SAMPLES, property_perc * 100,
                                                                                     theta))
    plt.legend(loc='upper left')

    plt.savefig('training_size{}_perc{}_theta{}_{}.png'.format(N_SAMPLES, property_perc * 100, theta, datetime))



    plt.figure(figsize=(15, 7))

    plt.subplot(1,2,1)
    plt.plot(range(N_EPOCHS), top_test_accuracies, label='Top')
    plt.plot(range(N_EPOCHS), rnd_test_accuracies, label='Random')
    plt.plot(range(N_EPOCHS), bottom_test_accuracies, label='Bottom')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='upper left')
    plt.title('Test accuracy of each model of the ensemble '
              '\n size of dataset: {} \n selected entropy = {}% | theta = {}'.format(N_SAMPLES, property_perc * 100, theta))

    plt.subplot(1,2,2)
    plt.plot(range(N_EPOCHS), top_ensemble_test_accuracies, label='Top')
    plt.plot(range(N_EPOCHS), rnd_ensemble_test_accuracies, label='Random')
    plt.plot(range(N_EPOCHS), bottom_ensemble_test_accuracies, label='Bottom')
    plt.xlabel('Ensemble', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.legend(loc='upper left')
    plt.title('Test accuracy of ensemble '
              '\n size of dataset: {} \n selected entropy = {}% | theta = {}'.format(N_SAMPLES, property_perc * 100, theta))

    plt.savefig('ensemble_accuracies_size{}_perc{}_theta{}_{}.png'.format(N_SAMPLES, property_perc * 100, theta, datetime))

    plt.figure()
    plt.plot(range(N_EPOCHS+1), top_lr, label='Top')
    plt.plot(range(N_EPOCHS+1), rnd_lr, label='Random')
    plt.plot(range(N_EPOCHS+1), bottom_lr, label='Bottom')
    plt.xlabel('Ensemble', fontsize=16)
    plt.ylabel('Lambda', fontsize=16)
    plt.title('Lambda of each model of the ensemble '
              '\n size of dataset: {} \n selected entropy = {}% | theta = {}'.format(N_SAMPLES, property_perc * 100, theta))
    plt.legend(loc='upper left')
    plt.savefig('ensemble_lr_size{}_perc{}_theta{}_{}.png'.format(N_SAMPLES, property_perc * 100, theta, datetime))
