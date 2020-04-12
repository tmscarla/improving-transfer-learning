import warnings

warnings.filterwarnings("ignore")
from experiments import *
import matplotlib.pyplot as plt
import numpy as np
from constants import *
from itertools import product
from utils import gen_mixed_plot
from training import train_baseline
from experiments import Experiment

# import remote_debugger

if __name__ == '__main__':
    # TODO Use 'SimpleBaselineBinaryNetTanh' if you want labels [-1,1]
    baseline = 'SimpleBaselineNet'
    dataset = 'CIFAR_10'
    distortion_type = 'AE_shift'
    property = 'entropy'
    property_perc = [0.25, 0.5]
    classes = range(0, 10)
    distortion_mean = [2, 3]

    for p in property_perc:
        experiment = Experiment(baseline=baseline, dataset=dataset, distortion_type=distortion_type,
                                mean=0, std=0, random=False, boosting_perc=1.0,
                                property='cross_entropy', property_perc=p, most=True, alternate=False,
                                classes=classes, classes_to_distort=None)
        experiment.run()
        experiment = Experiment(baseline=baseline, dataset=dataset, distortion_type=distortion_type,
                                mean=0, std=0, random=False, boosting_perc=1.0,
                                property='cross_entropy', property_perc=p, most=False, alternate=False,
                                classes=classes, classes_to_distort=None)
        experiment.run()
        experiment = Experiment(baseline=baseline, dataset=dataset, distortion_type=distortion_type,
                                mean=0, std=0, random=True, boosting_perc=1.0,
                                property='cross_entropy', property_perc=p, most=True, alternate=False,
                                classes=classes, classes_to_distort=None)
        experiment.run()


    exit()
    #
    # for p in property_perc:
    #     for d in distortion_mean:
    #         experiment = Experiment(baseline=baseline, dataset=dataset, distortion_type=distortion_type,
    #                                 mean=d, std=0, random=False, boosting_perc=1.0,
    #                                 property=property, property_perc=p, most=False, alternate=False,
    #                                 classes=classes, classes_to_distort=None)
    #         experiment.run()
    #
    # exit()

    ####### JACOBIAN ######

    # experiment = Experiment(baseline=baseline, dataset=dataset, distortion_type=distortion_type,
    #                         mean=5, std=0, random=False, boosting_perc=1.0,
    #                         property=None, property_perc=property_perc, most=None, alternate=None,
    #                         classes=classes, classes_to_distort=[1, 2, 3, 4], epochs_boosting=0)
    # experiment.run_jacobian(batch_size_jac=10)
    #
    # exit()

    ####### BOOSTING ######
    learning_rates = [0.001]
    boosting_percentages = [0.10, 0.25, 0.50]

    # for lr in learning_rates:
    #     for boosting_perc in boosting_percentages:
    #         experiment = Experiment(baseline=baseline, dataset=dataset, distortion_type=distortion_type,
    #                                 mean=0, std=0, random=False, boosting_perc=boosting_perc,
    #                                 property=None, property_perc=property_perc, most=None, alternate=None,
    #                                 classes=classes, classes_to_distort=None, epochs_boosting=5)
    #         experiment.run_boosting(lr=lr, loss='cross')


    ### RANDOM EXPERIMENTS ###
    for p in boosting_percentages:
        experiment = Experiment(baseline=baseline, dataset=dataset, distortion_type=distortion_type,
                                mean=0, std=0, random=True, boosting_perc=None,
                                property=None, property_perc=p, most=None, alternate=None,
                                classes=classes, classes_to_distort=None)
        experiment.run_boosting(lr=0.0, loss='cross')

    exit()

    # Random
    experiment = Experiment(baseline=baseline, dataset=dataset, distortion_type=distortion_type,
                            mean=0, std=0, random=True,
                            property=property, property_perc=property_perc, most=False,
                            classes=[3, 5], classes_to_distort=[5])

    experiment.run_recompute(total_epochs=50)

    exit()

    for params in list(product(baseline, dataset, distortion_type, std, most, property, property_perc)):
        baseline = params[0]
        dataset = params[1]
        distortion_type = params[2]
        std = params[3]
        most = params[4]
        property = params[5]
        property_perc = params[6]
        if not most and property_perc == 1.0:
            continue
        experiment = Experiment(baseline=baseline, dataset=dataset, distortion_type=distortion_type,
                                mean=15, std=0,
                                property=property, property_perc=property_perc, most=most,
                                classes=[3, 5], classes_to_distort=[3])
        path, writer = experiment.get_model_path_and_writer()
        experiment.run_recompute()

    baseline = ['SimpleNetBaseline']
    dataset = ['CIFAR_10']
    distortion_type = ['AWGN']
    std = [50]
    most = [True]
    property = ['entropy']
    property_perc = [0.2]

    for params in list(
            product(baseline, dataset, distortion_type, std, most, property, property_perc)):
        baseline = params[0]
        dataset = params[1]
        distortion_type = params[2]
        std = params[3]
        most = params[4]
        property = params[5]
        property_perc = params[6]
        if not most and property_perc == 1.0:
            continue
        experiment = Experiment(baseline=baseline, dataset=dataset, distortion_type=distortion_type,
                                distortion_amount=std,
                                property=property, property_perc=property_perc, most=most,
                                random=True, classes=[3, 5])
        experiment.run_recompute()
