import torch

BATCH_SIZE = 16
VALIDATION_PERCENTAGE = 0.20
TEST_PERCENTAGE = 0.20
RANDOM_SEED = 1234
PATIENCE = 15

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
