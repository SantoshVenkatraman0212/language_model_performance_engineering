'''
The sole purpose of this file is to check the versions of python, pytorch, cuda and it's availability
'''
# Importing necessary libraries
import sys
import torch
from config.settings import DEVICE

def main():
    '''
    Function that runs the version check print statements
    '''
    # Python version
    print(f'Python: {sys.version}')
    # Torch version
    print(f'PyTorch: {torch.__version__}')
    # CUDA availability
    print(f'CUDA availability: {torch.cuda.is_available()}')
    # CUDA version
    print(f'CUDA: {torch.version.cuda}')
    # GPU availability
    print(f'GPU availability: {torch.cuda.get_device_name() if DEVICE else None}')

if __name__ == '__main__':
    main()