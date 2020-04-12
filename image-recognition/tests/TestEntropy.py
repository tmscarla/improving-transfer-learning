import unittest
import numpy as np
from property import diversity_rearrangement
from collections import Counter
import scipy.stats as ss


class TestEntropy(unittest.TestCase):

    def discrete_normal_distribution(self, low, high):
        x = np.arange(low, high)
        xU, xL = x + 0.5, x - 0.5
        prob = ss.norm.cdf(xU, scale=3) - ss.norm.cdf(xL, scale=3)
        prob = prob / prob.sum()  # normalize the probabilities so their sum is 1
        nums = np.random.choice(x, size=10000, p=prob)
        return nums

    def test_diversity(self):
        X_train = np.random.randint(0, 1000, 10000)
        y_train = self.discrete_normal_distribution(0, 10)
        perc = 0.2
        min_per_class = 40
        if min_per_class > min(Counter(y_train).values()):
            min_per_class = min(Counter(y_train).values())

        X_train_sub, y_train_sub = diversity_rearrangement(X_train, y_train, perc, min_per_class)

        self.assertEqual(len(y_train_sub), len(y_train) * perc)
        self.assertEqual(min(Counter(y_train_sub).values()), min_per_class)


if __name__ == '__main__':
    unittest.main()
