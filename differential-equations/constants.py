import torch
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
dtype = torch.float

