from toy_experiment import *
import warnings

warnings.filterwarnings("ignore")
import datetime


def generate_toy_plots(train_accuracies, train_losses, val_accuracies, lr, boosting_perc, mode):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(train_accuracies)), train_accuracies, label='Train')
    plt.plot(range(len(val_accuracies)), val_accuracies, label='Validation')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    if mode == 'normal':
        plt.title('Accuracy | Boosting | LR: {} | Retaining: {}\n'.format(lr, boosting_perc))
    else:
        plt.title('Accuracy | Boosting | mode: {} | Retaining: {}\n'.format(mode, boosting_perc))
    plt.legend(loc='best')

    plt.subplot(1, 2, 2)
    plt.plot(range(len(train_losses)), train_losses, label='Train')
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Train loss - Boosting\n')
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    N_EPOCHS = 30
    property_perc = 0.7
    theta = 45.0
    N_SAMPLES = 1000
    toy = ToyExperiment(n_features=4, hidden_dim=16, n_classes=3, n_samples=N_SAMPLES, N_EPOCHS=N_EPOCHS,
                        alternate=False,
                        property='entropy', property_perc=property_perc, theta=theta, most=True,
                        convert_y=False)

    lr = 0.1
    boosting_perc = 1.0
    mode = 'entropy'
    train_acc, train_loss, val_acc, weights_boosting, _, _ = toy.toy_run_boosting(lr=lr, boosting_epochs=10,
                                                                                  boosting_perc=boosting_perc,
                                                                                  mode=mode)
    generate_toy_plots(train_acc, train_loss, val_acc, lr, boosting_perc, mode)
    print('--------------------------------------')
    lr = 0.1
    boosting_perc = 0.7
    mode = 'entropy'
    train_acc, train_loss, val_acc, weights_boosting, _, _ = toy.toy_run_boosting(lr=lr, boosting_epochs=10,
                                                                                  boosting_perc=boosting_perc,
                                                                                  mode=mode)
    generate_toy_plots(train_acc, train_loss, val_acc, lr, boosting_perc, mode)
    print('--------------------------------------')
    lr = 0.0
    boosting_perc = 1.0
    mode = 'normal'
    train_acc, train_loss, val_acc, weights_boosting, _, _ = toy.toy_run_boosting(lr=lr, boosting_epochs=10,
                                                                                  boosting_perc=boosting_perc,
                                                                                  mode=mode)
    generate_toy_plots(train_acc, train_loss, val_acc, lr, boosting_perc, mode)
    print('--------------------------------------')
    lr = 0.0
    boosting_perc = 0.7
    mode = 'normal'
    train_acc, train_loss, val_acc, weights_boosting, _, _ = toy.toy_run_boosting(lr=lr, boosting_epochs=10,
                                                                                  boosting_perc=boosting_perc,
                                                                                  mode=mode)
    generate_toy_plots(train_acc, train_loss, val_acc, lr, boosting_perc, mode)
