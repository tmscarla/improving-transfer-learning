import pandas as pd
from constants import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
    path = os.path.join(ROOT_DIR, 'tensorboard_csv', 'cross_entropy')
    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    train_top = pd.read_csv(path + '/run-CIFAR_10_AWGN-50_top0.1-tag-Accuracy_train.csv')
    train_bottom = pd.read_csv(path + '/run-CIFAR_10_AWGN-50_bottom0.1-tag-Accuracy_train.csv')
    train_random = pd.read_csv(path + '/run-CIFAR_10_AWGN-50_random0.1-tag-Accuracy_train.csv')
    plt.plot(train_top['Step'], train_top['Value'], label='top 10%')
    plt.plot(train_bottom['Step'], train_bottom['Value'], label='bottom 10%')
    plt.plot(train_random['Step'], train_random['Value'], label='random 10%')
    plt.legend(loc='upper left')
    plt.title('Train accuracy - Cross Entropy')
    plt.subplot(1,3,2)
    val_top = pd.read_csv(path + '/run-CIFAR_10_AWGN-50_top0.1-tag-Accuracy_val.csv')
    val_bottom = pd.read_csv(path + '/run-CIFAR_10_AWGN-50_bottom0.1-tag-Accuracy_val.csv')
    val_random = pd.read_csv(path + '/run-CIFAR_10_AWGN-50_random0.1-tag-Accuracy_val.csv')
    plt.plot(val_top['Step'], val_top['Value'], label='top 10%')
    plt.plot(val_bottom['Step'], val_bottom['Value'], label='bottom 10%')
    plt.plot(val_random['Step'], val_random['Value'], label='random 10%')
    plt.legend(loc='upper left')
    plt.title('Validation accuracy - Cross Entropy')

    plt.subplot(1,3,3)

    test_top = pd.read_csv(path+'/run-CIFAR_10_AWGN-50_top0.1-tag-Accuracy_test.csv')
    test_bottom = pd.read_csv(path+'/run-CIFAR_10_AWGN-50_bottom0.1-tag-Accuracy_test.csv')
    test_random = pd.read_csv(path+'/run-CIFAR_10_AWGN-50_random0.1-tag-Accuracy_test.csv')
    plt.plot(test_top['Step'], test_top['Value'], label='top 10%')
    plt.plot(test_bottom['Step'], test_bottom['Value'], label='bottom 10%')
    plt.plot(test_random['Step'], test_random['Value'], label='random 10%')
    plt.legend(loc='upper left')
    plt.title('Test accuracy - Cross Entropy')
    plt.savefig(os.path.join(ROOT_DIR, 'tensorboard_csv', 'cross_entropy', 'cross_entropy_recompute5.jpg'))
    plt.show()

