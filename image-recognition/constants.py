import os
import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

ROOT_DIR, tail = os.path.split(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = os.path.join(ROOT_DIR, 'harvard-thesis')
MODELS_DIR = 'models'
BASELINES_DIR = 'baselines'
RETRAINED_DIR = 'retrained'

DATA_DIR = 'data'
BATCH_SIZE = 64
RANDOM_SEED = 1234
PATIENCE = 15
VALIDATION_PERCENTAGE = 0.20

DSETS = ['USPS', 'CIFAR_100', 'CIFAR_10', 'MNIST']

CIFAR_10_LABELS = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer',
                   5: 'dog', 6: 'frog', 7: 'horse', 8: 'ship', 9: 'truck'}
