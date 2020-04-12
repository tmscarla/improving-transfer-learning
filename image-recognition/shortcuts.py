import time

from training import train_baseline
from utils import *
from utils import load_model_from_disk


def setup_finetuning(baseline, classes, dataset, distortion_type, mean, std, classes_to_distort=None):
    """
    Setup a baseline model trained on a pristine dataset and generate the distorted dataset based on the distortion
    type and parameters.
    :param baseline: name of the baseline model
    :param classes: array of labels
    :param dataset: dataset to be used for training the baseline
    :param distortion_type: type of distortion to generate the distorted one
    :param mean: mean of the noise
    :param std: standard deviation of the noise
    :param classes_to_distort: optional parameter to distort only a subset of the classes
    :return: model_clean, (X_train_noisy, X_test_noisy, X_train, X_test, y_train, y_test)
    """
    try:
        model_clean = load_model_from_disk(baseline, output_dim=len(classes))
    except FileNotFoundError:
        print('Baseline not found. Training...', end='')
        train_baseline(dataset, baseline, classes=classes, verbose=True)
        time.sleep(3)
        print('done!')
        model_clean = load_model_from_disk(baseline)
    try:
        X_train_noisy, X_test_noisy, X_train, X_test, y_train, y_test = load_data_from_disk(dataset,
                                                                                            distortion_type,
                                                                                            mean, std)
    except FileNotFoundError:
        print('Noisy data not found. Generating...', end='')
        gen_distorted_dataset(dataset, distortion_type, mean, std,
                              classes_to_distort=classes_to_distort)
        print('done!')
        time.sleep(3)
        X_train_noisy, X_test_noisy, X_train, X_test, y_train, y_test = load_data_from_disk(dataset,
                                                                                            distortion_type,
                                                                                            mean, std)
    # Filtering the selected classes
    if classes is not None:
        X_train, _ = select_classes(X_train, y_train, classes, convert_labels=True)
        X_test, _ = select_classes(X_test, y_test, classes, convert_labels=True)
        X_train_noisy, y_train = select_classes(X_train_noisy, y_train, classes, convert_labels=True)
        X_test_noisy, y_test = select_classes(X_test_noisy, y_test, classes, convert_labels=True)

    return model_clean, (X_train_noisy, X_test_noisy, X_train, X_test, y_train, y_test)
