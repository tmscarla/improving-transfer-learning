import numpy as np
import pandas as pd
from constants import ROOT_DIR
import matplotlib.pyplot as plt

if __name__ == '__main__':
    file_name = 'CIFAR_10_AE_shift_bottom_0.25_train.csv'
    df = pd.read_csv(ROOT_DIR + '/plots/csv/{}'.format(file_name))
    steps_b_train, values_b_train = df['Step'].to_numpy(), df['Value'].to_numpy()

    file_name = 'CIFAR_10_AE_shift_random_0.25_train.csv'
    df = pd.read_csv(ROOT_DIR + '/plots/csv/{}'.format(file_name))
    steps_r_train, values_r_train = df['Step'].to_numpy(), df['Value'].to_numpy()

    file_name = 'CIFAR_10_AE_shift_top_0.25_train.csv'
    df = pd.read_csv(ROOT_DIR + '/plots/csv/{}'.format(file_name))
    steps_t_train, values_t_train = df['Step'].to_numpy(), df['Value'].to_numpy()

    file_name = 'CIFAR_10_AE_shift_bottom_0.25_test.csv'
    df = pd.read_csv(ROOT_DIR + '/plots/csv/{}'.format(file_name))
    steps_b_test, values_b_test = df['Step'].to_numpy(), df['Value'].to_numpy()

    file_name = 'CIFAR_10_AE_shift_random_0.25_test.csv'
    df = pd.read_csv(ROOT_DIR + '/plots/csv/{}'.format(file_name))
    steps_r_test, values_r_test = df['Step'].to_numpy(), df['Value'].to_numpy()

    file_name = 'CIFAR_10_AE_shift_top_0.25_test.csv'
    df = pd.read_csv(ROOT_DIR + '/plots/csv/{}'.format(file_name))
    steps_t_test, values_t_test = df['Step'].to_numpy(), df['Value'].to_numpy()

    fig = plt.figure(figsize=(15, 5))
    st = fig.suptitle('Accuracy trend of a baseline pre-trained on CIFAR 10 dataset\n'
                      'and finetuned on a distorted version with embedding shift = 0.0\n'
                      'Samples selected according to error-driven criterion')
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.set_title('Train')
    ax1.plot(steps_t_train, values_t_train, linewidth=2, color='#cc0000', label='Top 25%')
    ax1.plot(steps_b_train, values_b_train, linewidth=2, color='#cc0000', label='Bottom 25%', linestyle='--')
    ax1.plot(steps_r_train, values_r_train, linewidth=2, color='#3366ff', label='Random 25%', linestyle='dotted')
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend()
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.set_title('Test')
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.plot(steps_t_test, values_t_test, linewidth=2, color='#cc0000', label='Top 25%')
    ax2.plot(steps_b_test, values_b_test, linewidth=2, color='#cc0000', label='Bottom 25%', linestyle='--')
    ax2.plot(steps_r_test, values_r_test, linewidth=2, color='#3366ff', label='Random 25%', linestyle='dotted')
    ax2.legend()

    fig.tight_layout()

    st.set_y(0.95)
    fig.subplots_adjust(top=0.82)

    plt.savefig('img.png')
    fig.show()
