from sklearn.model_selection import StratifiedShuffleSplit

from downloads import load_dataset
from models import *
from datasets import image_preprocessing, gen_distorted_dataset
import pickle
import numpy as np
from matplotlib import pyplot as plt
from constants import *
from nets.resnet import ResNet18
from sklearn.decomposition import PCA
from matplotlib import animation, rc
from matplotlib.colors import ListedColormap
from datasets import get_data_loader
from tqdm import tqdm


plt.rcParams.update({'figure.max_open_warning': 0})
rc('animation', html='html5')


def convert_labels(y, old_labels, new_labels):
    """
    Convert labels of a numpy array using an implicit mapping from old_labels to new_labels.
    :param y: numpy array of labels
    :param old_labels: e.g. [0, 1]
    :param new_labels: e.g. [-1, 1]
    :return: y_converted
    """
    assert len(old_labels) == len(new_labels)
    if set(y) == set(new_labels):
        return y
    mapping = dict(zip(old_labels, new_labels))
    y_converted = [mapping[y_i] for y_i in y]
    return np.array(y_converted)


def dataset_split(X, y, perc=VALIDATION_PERCENTAGE, random_state=RANDOM_SEED, return_data='samples'):
    """
    Given two arrays of samples and label X and y, perform a random splitting in train and validation sets.
    :param X: numpy array of samples
    :param y: numpy array of labels
    :param perc: percentage of validation set
    :param random_state: random state of the splitter
    :param return_data: if True, return DataLoader objects instead of numpy arrays
    :return: (train_loader, val_loader) or (X_train, y_train), (X_val, y_val) or train_idx, val_idx
    """
    assert 0 <= perc <= 1

    sss = StratifiedShuffleSplit(n_splits=1, test_size=perc, random_state=random_state)
    train_idxs, valid_idxs = next(sss.split(X, y))

    X_train, X_valid = X[train_idxs], X[valid_idxs]
    y_train, y_valid = y[train_idxs], y[valid_idxs]

    if return_data == 'data_loader':
        return get_data_loader(X_train, y_train), get_data_loader(X_valid, y_valid)
    elif return_data == 'samples':
        return (X_train, y_train), (X_valid, y_valid)
    elif return_data == 'indices':
        return train_idxs, valid_idxs


def visualize_decision_boundary(epoch_features, y, epoch_entropy_indices, interval=200, file_path='db.gif'):
    """
    Visualize the evolution of the decision boundary of the network.
    :param epoch_features: numpy array of shape (epochs, samples, features)
    :param y: numpy array of labels of length=samples
    :param epoch_entropy_indices: numpy array of shape (epochs,)
    :param interval: interval of time (in ms) of each image of the animation
    :return:
    """
    # Use t-sne to reduce dimensionality for each epoch
    tsne_results = []

    for i in tqdm(range(len(epoch_features)), desc='TSNE'):
        # tsne = MultiTSNE(n_jobs=4, n_components=2, verbose=0, perplexity=70, n_iter=400)
        # tsne_result = tsne.fit_transform(np.array(epoch_features[i]))
        pca = PCA(n_components=2)
        tsne_result = pca.fit_transform(np.array(epoch_features[i]))
        tsne_results.append(tsne_result)

    # Setup
    tsne_results = np.array(tsne_results)
    fig, ax = plt.subplots()
    fig.set_dpi(80)
    fig.set_size_inches(10, 6)
    ax.set_xlim((np.min(tsne_results[:, :, 0])-0.2, np.max(tsne_results[:, :, 0])+0.2))
    ax.set_ylim((np.min(tsne_results[:, :, 1])-0.2, np.max(tsne_results[:, :, 1])+0.2))
    ax.set_title("Decision boundaries evolution - TSNE\n entropy | selected {}%".format(int(0.1 * 100)))

    colors = ListedColormap(['r', 'b'])
    features = ax.scatter([], [], marker='o', c=[], s=40, alpha=0.3)
    scatter_db = ax.scatter([], [], marker='o', s=40, alpha=1.0, edgecolor='k', color='green')

    line_db, = ax.plot([], [], '-', lw=3, color='green', marker='s', alpha=0.9)

    # ax.legend((features.legend_elements()[0][0], features.legend_elements()[0][1]), ('a', 'b'))

    def animate_func(i):
        # Update positions and labels
        features.set_offsets(tsne_results[i])
        features.set_array(y[i])
        ax.set_xlabel('Epoch = {}'.format(i))

        if epoch_entropy_indices is not None:
            db = tsne_results[i][epoch_entropy_indices[i][:40]]
            scatter_db.set_offsets(db)
            # spline = interp1d(db[:, 0], db[:, 1], kind='cubic')
            # line_db.set_data(db[:, 0], spline(db[:, 0]))

        return features, scatter_db

    # Create and save animation
    print('Creating gif...', end='')
    anim = animation.FuncAnimation(fig, animate_func, frames=len(epoch_features), interval=interval, blit=True)
    anim.save(file_path, dpi=80)
    print('done!')


def get_random_subset(X_train_noisy, y_train, perc, return_indices=False):
    """
    Pick a random subset of the samples
    :param X_train_noisy: numpy array of noisy samples
    :param y_train: numpy array of labels
    :param perc: percentage of samples to be selected
    :param return_indices: if True, return indices instead of samples
    :return: indices | (X_train_noisy_subset, y_train_subset)
    """
    total_len = X_train_noisy.shape[0]
    subset_len = int(total_len * perc)
    indices = np.random.choice(total_len, subset_len, replace=False)
    X_train_noisy_subset = X_train_noisy[indices]
    y_train_subset = y_train[indices]

    if return_indices:
        return indices
    else:
        return X_train_noisy_subset, y_train_subset


def show_rand_img(data_path, X, y, verbose=True):
    """
    Plot a random image from the
    :param data_path: path for the folder that contains X and y
    :param X: numpy array of samples
    :param y: numpy array of labels
    :param verbose: add verbosity
    """
    X = np.load(os.path.join(data_path, X))
    X = image_preprocessing(X)
    y = np.load(os.path.join(data_path, y))

    np.random.seed(int(time.time()))
    i = np.random.randint(0, len(X))
    img = X[i]

    plt.imshow(img)
    plt.show()
    if verbose:
        print(y[i])
        plt.savefig('sample_img_{}_label={}.png'.format(i,y[i]))


def load_model_from_disk(model_name, output_dim=''):
    """
    Create a new model of type model_name and load its baseline parameters from disk.
    :param model_name: name of the baseline model to be loaded
    :return: net: model trained with parameters loaded from disk
    """
    path = os.path.join(ROOT_DIR, MODELS_DIR, BASELINES_DIR, model_name + '.pt')
    if model_name == 'SimpleBaselineNet':
        net = SimpleBaselineNet()
    elif model_name == 'SimpleBaselineNet_{}'.format(output_dim):
        net = SimpleBaselineNet()
    elif model_name == 'ACNBaselineNet':
        net = ACNBaselineNet()
    elif model_name == 'ResNetBaseline':
        net = ResNet18()
    elif model_name == 'FFNetBaseline':
        net = FFSimpleNet()
    elif model_name == 'SimpleBaselineBinaryNet':
        net = SimpleBaselineBinaryNet()
    elif model_name == 'SimpleBaselineBinaryNetTanh':
        net = SimpleBaselineBinaryNet(activation='tanh')
    elif model_name == 'FFBinaryNet':
        net = FFBinaryNet()
    elif model_name == 'SimplerBaselineBinaryNetTanh':
        net = SimpleBaselineBinaryNet(activation='tanh', num_conv=32, num_ff=32)
    else:
        raise RuntimeError("Model name '{}' not in list!".format(model_name))

    net.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage))
    return net


def load_data_from_disk(dataset, distortion_type, mean, std):
    """
    Load clean and noisy data from disk.
    :param dataset: name of the dataset
    :param distortion_type: ['AWGN', 'blur']
    :param mean: mean of the noise
    :param std: standard deviation of the noise
    :return: X_train_noisy, X_test_noisy, X_train, X_test, y_train, y_test
    """
    # Load noisy dataset
    data_path = os.path.join(ROOT_DIR, DATA_DIR, dataset, distortion_type + '-m=' + str(mean) + '-std=' + str(std))

    X_train_noisy = np.load(os.path.join(data_path, 'X_train_noisy.npy'))
    X_test_noisy = np.load(os.path.join(data_path, 'X_test_noisy.npy'))

    # Load clean dataset
    X_train = np.load(os.path.join(data_path, 'X_train.npy'))
    y_train = np.load(os.path.join(data_path, 'y_train.npy'))
    X_test = np.load(os.path.join(data_path, 'X_test.npy'))
    y_test = np.load(os.path.join(data_path, 'y_test.npy'))

    # Flatten for the dataloader
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    return X_train_noisy, X_test_noisy, X_train, X_test, y_train, y_test


def load_dataloaders_from_dataset(dataset):
    (X_train, y_train), (X_test, y_test) = load_dataset(dataset)

    # Scale pixels values
    X_train, X_mean, X_std = image_preprocessing(X_train, scale_only=False)
    X_test, _, _ = image_preprocessing(X_test, seq_mean=X_mean, seq_std=X_std, scale_only=False)

    # Flatten for the dataloader
    y_train = y_train.flatten()
    y_test = y_test.flatten()

    # Stratified split of training and validation
    sss = StratifiedShuffleSplit(n_splits=1, test_size=VALIDATION_PERCENTAGE, random_state=RANDOM_SEED)
    train_idx, val_idx = next(sss.split(X_train, y_train))
    (X_train, X_valid) = X_train[train_idx], X_train[val_idx]
    (y_train, y_valid) = y_train[train_idx], y_train[val_idx]

    # Generating data loaders
    train_dl = get_data_loader(X_train, y_train)
    val_dl = get_data_loader(X_valid, y_valid, shuffle=False)
    test_dl = get_data_loader(X_test, y_test, shuffle=False)

    return train_dl, val_dl, test_dl


def get_arrays_from_data_loader(data_loader):
    """
    Generate numpy arrays of samples and labels from a DataLoader object.
    :param data_loader: the DataLoader object
    :return: X, y: numpy arrays
    """
    dataset = data_loader.dataset
    X = dataset.X
    y = dataset.y

    # Make it coherent with the reshaping of dataloader generation
    X = np.reshape(X, [X.shape[0], 32, 32, 3])
    return X, y


def save_log(list, path, object_name):
    """
    Save log file in /runs folder
    :param list:
    :param path:
    :param object_name:
    """
    filename = path.replace('runs/', '') + '_' + object_name
    pckl_file = open(os.path.join(path, filename) + '.txt', "wb")
    pickle.dump(list, pckl_file)
    pckl_file.close()


def select_classes(X, y, classes, convert_labels=True, new_labels=None):
    """
    Keep only the samples of X and y which labels match classes.
    :param X: numpy array of samples
    :param y: numpy array of labels
    :param classes: classes to be selected among the classes of the dataset
    :param convert_labels: if True, convert class labels. E.g. classes = [3,5] --> {3: 0, 5: 1}
    :param new_labels: if convert_labels is True, use new_labels for the mapping
    :return: X_subset, y_subset: subset of X and y of selected classes only
    """
    assert len(X) == len(y)
    if classes is None:
        return X, y

    idx = np.isin(y, classes)
    X_subset, y_subset = X[idx.flatten()], y[idx]

    if convert_labels:
        if new_labels is not None:
            mapping = dict(zip(classes, new_labels))
        else:
            mapping = dict(zip(classes, range(len(classes))))
        y_subset = np.array([mapping[y] for y in y_subset])

    return X_subset, y_subset


def gen_plots(path):
    all_losses = []
    all_accuracies = []

    # Losses
    for str in ['_train-loss', '_val-loss']:
        filename = path.replace('runs/', '') + str
        losses = pickle.load(open(os.path.join(path, filename) + '.txt', 'rb'))
        all_losses.append(losses)
        plt.figure()
        plt.plot(losses)
        plt.title(filename.replace('_', ' '))
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(path, filename) + '.png')

    # Accuracies
    for str in ['_train-acc', '_val-acc', '_test-acc']:
        filename = path.replace('runs/', '') + str
        accuracies = pickle.load(open(os.path.join(path, filename) + '.txt', 'rb'))
        all_accuracies.append(accuracies)
        plt.figure()
        plt.plot(accuracies)
        plt.title(filename.replace('_', ' '))
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.savefig(os.path.join(path, filename) + '.png')

    # Generate overlapping plots
    plt.figure()
    for losses in all_losses:
        plt.plot(losses)
    filename = path.replace('runs/', '').replace('_', ' ') + ' Loss'
    plt.title(filename)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(os.path.join(path, filename) + '.png')

    plt.figure()
    for accuracies in all_accuracies:
        plt.plot(accuracies)
    filename = path.replace('runs/', '').replace('_', ' ') + ' Accuracy'
    plt.title(filename)
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation', 'Test'], loc='upper left')
    plt.savefig(os.path.join(path, filename) + '.png')


def gen_mixed_plot(dataset, distortion_amount, distortion_type, entropy_percentage, other=''):
    # just check if the all configuration is in the property_perc array
    for amt in distortion_amount:
        for ent in entropy_percentage:
            if ent == 1.0:
                # skip loop
                continue
            path_top = os.path.join('runs',
                                    dataset + '_' + distortion_type + '-' + str(amt) + '_top' + str(ent) + other)
            path_bottom = os.path.join('runs',
                                       dataset + '_' + distortion_type + '-' + str(amt) + '_bottom' + str(ent) + other)
            path_all = os.path.join('runs', dataset + '_' + distortion_type + '-' + str(amt) + '_top1.0' + other)
            for acc in ['_train-acc', '_val-acc', '_test-acc']:
                all_accuracies = []
                filename_top = path_top.replace('runs/', '') + acc
                filename_bottom = path_bottom.replace('runs/', '') + acc
                filename_all = path_all.replace('runs/', '') + acc
                accuracies_top = pickle.load(open(os.path.join(path_top, filename_top) + '.txt', 'rb'))
                accuracies_bottom = pickle.load(open(os.path.join(path_bottom, filename_bottom) + '.txt', 'rb'))
                accuracies_all = pickle.load(open(os.path.join(path_all, filename_all) + '.txt', 'rb'))
                plt.rcParams.update({'font.size': 26})
                all_accuracies.append(accuracies_top)
                all_accuracies.append(accuracies_bottom)
                all_accuracies.append(accuracies_all)
                plt.figure(figsize=(40, 20))
                for accuracies in all_accuracies:
                    plt.plot(accuracies)
                filename = dataset + '_' + distortion_type + '-' + str(amt) + other + '_test_acc'
                path = os.path.join('plots', dataset + '_' + distortion_type + '-' + str(amt) + other)
                plt.title(filename)
                plt.ylabel('Accuracy')
                plt.xlabel('Epoch')
                plt.legend(['top ' + str(ent), 'bottom ' + str(ent), 'all'], loc='upper left')
                if not os.path.exists(path):
                    os.makedirs(path)
                plt.savefig(os.path.join(path, filename) + '.png')

