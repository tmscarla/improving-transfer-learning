import unittest
import numpy as np
from property import diversity_rearrangement
from collections import Counter
import scipy.stats as ss
from experiments import *
from property import *


class TestExperiments(unittest.TestCase):

    def test_get_random_subset(self):
        baseline = 'SimpleNetBaseline'
        dataset = 'CIFAR_10'
        distortion_type = 'AWGN'
        distortion_amount = 15
        percentage = 0.25
        exp = Experiment(baseline=baseline, dataset=dataset, distortion_type=distortion_type,
                         distortion_amount=distortion_amount, property_perc=percentage, per_class=False, most=True,
                         diversity=False)
        net = load_model_from_disk(dataset)
        X_train_noisy, X_test_noisy, X_train, X_test, y_train, y_test = load_data_from_disk(dataset,
                                                                                            distortion_type,, distortion_amount

        X_sub_top, y_sub_top = get_samples_by_entropy(net, X_train_noisy, y_train, percentage, True)
        X_sub_rand, y_sub_rand = exp.get_random_subset(X_train_noisy, y_train)
        X_sub_bottom, y_sub_bottom = get_samples_by_entropy(net, X_train_noisy, y_train, percentage, False)

        entropies_top = compute_entropy(net, X_sub_top, y_sub_top, indices=False)
        entropies_rand = compute_entropy(net, X_sub_rand, y_sub_rand, indices=False)
        entropies_bottom = compute_entropy(net, X_sub_bottom, y_sub_bottom, indices=False)

        entropy_mean_top = np.mean(entropies_top)
        entropy_mean_rand = np.mean(entropies_rand)
        entropy_mean_bottom = np.mean(entropies_bottom)

        assert entropy_mean_top > entropy_mean_rand > entropy_mean_bottom


if __name__ == '__main__':
    unittest.main()
