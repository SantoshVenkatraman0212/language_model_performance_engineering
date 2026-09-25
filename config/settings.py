'''
This file has the hyperparams, and shared variables
'''

# Importing necessary libraries
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'