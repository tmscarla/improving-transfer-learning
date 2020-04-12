from property import *
from utils import *
import matplotlib.pyplot as plt
from experiments import *


def plot_class_hist(y, title):
    plt.figure()
    plt.title(title)
    plt.hist(y)
    plt.show()


def generate_boosting_plots(epochs, train_accuracies, train_losses, val_accuracies, weights,
                            learning_rate_collection,
                            epsilon_collection, lr, save=True):
    # plt.figure()
    # plt.plot(range(epochs), train_accuracies)
    # plt.xlabel('Epochs', fontsize=16)
    # plt.ylabel('Accuracy', fontsize=16)
    # plt.title('Train accuracy - simple NN Boosting\n')
    # if save: plt.savefig(ROOT_DIR + '/plots/train_accuracy.png')
    # else: plt.show()
    #
    # plt.figure()
    # plt.plot(range(epochs), train_losses)
    # plt.xlabel('Epochs', fontsize=16)
    # plt.ylabel('Loss', fontsize=16)
    # plt.title('Train loss - simple NN Boosting\n')
    # if save: plt.savefig('/plots/train_loss.png')
    # else: plt.show()
    #
    # plt.figure()
    # plt.plot(range(epochs), train_losses)
    # plt.xlabel('Epochs', fontsize=16)
    # plt.ylabel('Accuracy', fontsize=16)
    # plt.title('Validation accuracy - simple NN Boosting\n')
    # if save: plt.savefig('/plots/val_accuracy.png')
    # else: plt.show()

    n_uniques = []
    for w in weights:
        n_uniques.append(len(np.unique(list(w.values()))))

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.hist(weights[0].values())
    plt.title('Weights distribution - starting point\n')

    plt.subplot(1, 3, 2)
    plt.hist(weights[len(weights)//2].values())
    plt.title('Weights distribution - epoch {}\n'.format(len(weights)//2))

    plt.subplot(1, 3, 3)
    plt.hist(weights[-1].values())
    plt.title('Weights distribution - epoch {}\n'.format(epochs))

    if not lr: lr = ''
    if save: plt.savefig('weights_lr={}.png'.format(lr))
    else: plt.show()


if __name__ == '__main__':
    net = load_model_from_disk('CIFAR_10')
    distortion_type = 'AWGN'
    std = 50
    mean = 15
    entropy_percentage = 0.25
    try:
        x_train_noisy, x_test_noisy, x_train, x_test, y_train, y_test = load_data_from_disk('CIFAR_10',
                                                                                            distortion_type, std)
    except FileNotFoundError:
        print('Noisy data not found. Generating...')
        gen_distorted_dataset('CIFAR_10', distortion_type, mean, std)
        x_train_noisy, x_test_noisy, x_train, x_test, y_train, y_test = load_data_from_disk('CIFAR_10',
                                                                                            distortion_type, std)

    x_train_noisy = image_preprocessing(x_train_noisy, scale_only=False)
    x_test_noisy = image_preprocessing(x_test_noisy, scale_only=False)
    x_test = image_preprocessing(x_test, scale_only=False)

    n_labels = len(np.unique(y_train))
    entropies = compute_samples_property(net, x_train_noisy, y_train, n_labels, indices=False)
    p25 = np.percentile(entropies, 25)
    p75 = np.percentile(entropies, 75)
    test_entropies = compute_samples_property(net, x_test_noisy, y_test, n_labels, indices=False)

    plt.figure()
    plt.title('Entropy distribution on %s %d' % (distortion_type, std))
    plt.hist(entropies, bins=50)
    plt.axvline(p25, color='r')
    plt.axvline(p75, color='r')
    plt.show()

    plt.figure()
    plt.title('Test entropy distribution on %s %d' % (distortion_type, std))
    plt.hist(test_entropies, bins=50)
    plt.show()

    cross_entropies = compute_samples_property(net, x_train_noisy, y_train, n_labels, indices=False)
    p25 = np.percentile(cross_entropies, 25)
    p75 = np.percentile(cross_entropies, 75)
    test_cross_entropies = compute_samples_property(net, x_test_noisy, y_test, n_labels, indices=False)

    plt.figure()
    plt.title('Cross Entropy distribution on %s %d' % (distortion_type, std))
    plt.hist(cross_entropies, bins=50)
    plt.axvline(p25, color='r')
    plt.axvline(p75, color='r')
    plt.plot(p25, 0, 'ro')
    plt.plot(p75, 0, 'ro')
    plt.show()

    plt.figure()
    plt.title('Test cross entropy distribution on %s %d' % (distortion_type, std))
    plt.hist(test_cross_entropies, bins=50)
    plt.show()

    ######## NORMAL SAMPLING ########

    # x_train_noisy_top, y_train_top = get_samples_by_entropy(net, x_train_noisy, y_train,
    #                                                               property_perc, most=True)
    # x_train_noisy_bottom, y_train_bottom = get_samples_by_entropy(net, x_train_noisy, y_train,
    #                                                               property_perc, most=False)
    # x_train_noisy_random, y_train_random = get_random_subset(x_train_noisy, y_train, property_perc, per_class=False)
    #
    #
    # top_entropies = compute_entropy(net, x_train_noisy_top, y_train_top, n_labels, indices=False)
    # bottom_entropies = compute_entropy(net, x_train_noisy_bottom, y_train_bottom, n_labels, indices=False)
    # random_entropies = compute_entropy(net, x_train_noisy_random, y_train_random, n_labels, indices=False)
    #
    # plt.figure()
    # plt.title('Entropy distribution on {} {} top {}'.format(distortion_type, distortion_amount, property_perc) )
    # plt.hist(top_entropies, bins=50)
    # plt.show()
    # print('Entropy mean on {} {} top {}: {}'.format(distortion_type, distortion_amount, property_perc, np.mean(top_entropies)))
    # print('Entropy sum on {} {} top {}: {}'.format(distortion_type, distortion_amount, property_perc, np.sum(top_entropies)))
    #
    #
    # plt.figure()
    # plt.title('Entropy distribution on {} {} bottom {}'.format(distortion_type, distortion_amount, property_perc) )
    # plt.hist(bottom_entropies, bins=50)
    # plt.show()
    # print('Entropy mean on {} {} bottom {}: {}'.format(distortion_type, distortion_amount, property_perc, np.mean(bottom_entropies)))
    # print('Entropy sum on {} {} bottom {}: {}'.format(distortion_type, distortion_amount, property_perc, np.sum(bottom_entropies)))
    #
    #
    # plt.figure()
    # plt.title('Entropy distribution on {} {} random {}'.format(distortion_type, distortion_amount, property_perc) )
    # plt.hist(random_entropies, bins=50)
    # plt.show()
    # print('Entropy mean on {} {} random {}: {}'.format(distortion_type, distortion_amount, property_perc, np.mean(random_entropies)))
    # print('Entropy sum on {} {} random {}: {}'.format(distortion_type, distortion_amount, property_perc, np.sum(random_entropies)))
    #
    # x_train_noisy_top, y_train_top = get_samples_by_entropy(net, x_train_noisy, y_train,
    #                                                         property_perc, cross_entropy = True, most=True)
    # x_train_noisy_bottom, y_train_bottom = get_samples_by_entropy(net, x_train_noisy, y_train,
    #                                                               property_perc, cross_entropy = True, most=False)
    #
    # top_cross_entropies = compute_entropy(net, x_train_noisy_top, y_train_top, n_labels, cross_entropy=True, indices=False)
    # bottom_cross_entropies = compute_entropy(net, x_train_noisy_bottom, y_train_bottom, n_labels, cross_entropy=True, indices=False)
    # random_cross_entropies = compute_entropy(net, x_train_noisy_random, y_train_random, n_labels, cross_entropy=True, indices=False)
    #
    # plt.figure()
    # plt.title('Cross entropy distribution on {} {} top {}'.format(distortion_type, distortion_amount, property_perc))
    # plt.hist(top_cross_entropies, bins=50)
    # plt.show()
    # print('Cross entropy mean on {} {} top {}: {}'.format(distortion_type, distortion_amount, property_perc, np.mean(top_cross_entropies)))
    # print('Cross entropy sum on {} {} top {}: {}'.format(distortion_type, distortion_amount, property_perc, np.sum(top_cross_entropies)))
    #
    # plt.figure()
    # plt.title('Cross entropy distribution on {} {} bottom {}'.format(distortion_type, distortion_amount, property_perc))
    # plt.hist(bottom_cross_entropies, bins=50)
    # plt.show()
    # print('Cross entropy mean on {} {} bottom {}: {}'.format(distortion_type, distortion_amount, property_perc, np.mean(bottom_cross_entropies)))
    # print('Cross entropy sum on {} {} bottom {}: {}'.format(distortion_type, distortion_amount, property_perc, np.sum(bottom_cross_entropies)))
    #
    # plt.figure()
    # plt.title('Cross entropy distribution on {} {} random {}'.format(distortion_type, distortion_amount, property_perc))
    # plt.hist(random_cross_entropies, bins=50)
    # plt.show()
    # print('Cross entropy mean on {} {} random {}: {}'.format(distortion_type, distortion_amount, property_perc, np.mean(random_cross_entropies)))
    # print('Cross entropy sum on {} {} random {}: {}'.format(distortion_type, distortion_amount, property_perc, np.sum(random_cross_entropies)))

    # TODO THE FOLLOWING PLOT IS NOT CORRECT! MUST REDO SAMPLING FOR CROSS ENTROPY
    ######## PER CLASS ########
    # x_train_noisy_top, y_train_top = get_samples_by_entropy_per_class(net, x_train_noisy, y_train,
    #                                                               property_perc, most=True)
    # x_train_noisy_bottom, y_train_bottom = get_samples_by_entropy_per_class(net, x_train_noisy, y_train,
    #                                                               property_perc, most=False)
    # x_train_noisy_random, y_train_random = get_random_subset(x_train_noisy, y_train, property_perc, per_class=True)
    #
    #
    # top_entropies = compute_entropy(net, x_train_noisy_top, y_train_top, n_labels, indices=False)
    # bottom_entropies = compute_entropy(net, x_train_noisy_bottom, y_train_bottom, n_labels, indices=False)
    # random_entropies = compute_entropy(net, x_train_noisy_random, y_train_random, n_labels, indices=False)
    #
    # plt.figure()
    # plt.title('Entropy distribution on {} {} top {}'.format(distortion_type, distortion_amount, property_perc) )
    # plt.hist(top_entropies, bins=50)
    # plt.show()
    # print('Entropy mean on {} {} top {}: {}'.format(distortion_type, distortion_amount, property_perc, np.mean(top_entropies)))
    #
    #
    # plt.figure()
    # plt.title('Entropy distribution on {} {} bottom {}'.format(distortion_type, distortion_amount, property_perc) )
    # plt.hist(bottom_entropies, bins=50)
    # plt.show()
    # print('Entropy mean on {} {} bottom {}: {}'.format(distortion_type, distortion_amount, property_perc, np.mean(bottom_entropies)))
    #
    #
    # plt.figure()
    # plt.title('Entropy distribution on {} {} random {}'.format(distortion_type, distortion_amount, property_perc) )
    # plt.hist(random_entropies, bins=50)
    # plt.show()
    # print('Entropy mean on {} {} random {}: {}'.format(distortion_type, distortion_amount, property_perc, np.mean(random_entropies)))
    #
    #
    # top_cross_entropies = compute_entropy(net, x_train_noisy_top, y_train_top, n_labels, cross_entropy=True, indices=False)
    # bottom_cross_entropies = compute_entropy(net, x_train_noisy_bottom, y_train_bottom, n_labels, cross_entropy=True, indices=False)
    # random_cross_entropies = compute_entropy(net, x_train_noisy_random, y_train_random, n_labels, cross_entropy=True, indices=False)
    #
    # plt.figure()
    # plt.title('Cross Entropy distribution on {} {} top {}'.format(distortion_type, distortion_amount, property_perc))
    # plt.hist(top_cross_entropies, bins=50)
    # plt.show()
    # print('Cross entropy mean on {} {} top {}: {}'.format(distortion_type, distortion_amount, property_perc, np.mean(top_cross_entropies)))
    #
    # plt.figure()
    # plt.title('Cross Entropy distribution on {} {} bottom {}'.format(distortion_type, distortion_amount, property_perc))
    # plt.hist(bottom_cross_entropies, bins=50)
    # plt.show()
    # print('Cross entropy mean on {} {} bottom {}: {}'.format(distortion_type, distortion_amount, property_perc, np.mean(bottom_cross_entropies)))
    #
    # plt.figure()
    # plt.title('Cross Entropy distribution on {} {} random {}'.format(distortion_type, distortion_amount, property_perc))
    # plt.hist(random_cross_entropies, bins=50)
    # plt.show()
    # print('Cross entropy mean on {} {} random {}: {}'.format(distortion_type, distortion_amount, property_perc, np.mean(random_cross_entropies)))
