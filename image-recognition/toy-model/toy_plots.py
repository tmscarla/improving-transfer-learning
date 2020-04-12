import matplotlib.pyplot as plt
import numpy as np


def generate_simple_boosting_plots(epochs, train_accuracies, train_losses, val_accuracies, weights,
                                   learning_rate_collection,
                                   epsilon_collection):
    plt.figure()
    plt.plot(range(epochs), train_accuracies)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Train accuracy - simple NN Boosting\n')
    plt.show()

    plt.figure()
    plt.plot(range(epochs), train_losses)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.title('Train loss - simple NN Boosting\n')
    plt.show()

    plt.figure()
    plt.plot(range(epochs), train_losses)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Accuracy', fontsize=16)
    plt.title('Validation accuracy - simple NN Boosting\n')
    plt.show()

    exit()

    n_uniques = []
    for w in weights:
        n_uniques.append(len(np.unique(list(w.values()))))

    plt.figure()
    plt.plot(range(len(n_uniques)), n_uniques)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Number of unique', fontsize=16)
    plt.title('Number of uniques in the weights vector\n')

    plt.figure()
    plt.plot(range(len(learning_rate_collection)), learning_rate_collection)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Lambda', fontsize=16)
    plt.title('Lambda - simple NN Boosting\n')

    plt.figure()
    plt.plot(range(len(epsilon_collection)), epsilon_collection)
    plt.xlabel('Epochs', fontsize=16)
    plt.ylabel('Epsilon', fontsize=16)
    plt.title('Epsilon - simple NN Boosting\n')

    plt.figure(figsize=(15,5))
    plt.subplot(1, 3, 1)
    plt.hist(weights[0].values())
    plt.title('Weights distribution - starting point\n')

    plt.subplot(1, 3, 2)
    plt.hist(weights[10].values())
    plt.title('Weights distribution - epoch {}\n'.format(10))

    plt.subplot(1, 3, 3)
    plt.hist(weights[epochs - 1].values())
    plt.title('Weights distribution - epoch {}\n'.format(epochs))

    plt.show()
