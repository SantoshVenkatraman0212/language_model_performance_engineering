'''
This file implements naive softmax operation on a 2-D Tensor, and depicts numerical overflow
'''
# Importing necessary libraries
import torch

def naive_softmax(x: torch.tensor) -> torch.tensor:
    '''
    This function implements naive softmax (without global maxima diff in exponent)
    on a 2-D tensor
    '''
    # Creates a tensor of every value of X raised to exp
    num = torch.exp(x)
    # Performs a row-wise sum of exp(X)
    # keepdims ensures that denom tensor is 2-D (2, 1) which is required for tensor broadcasting
    denom = torch.sum(num, dim = -1, keepdims = True)
    naive_sm = num / denom

    return naive_sm

def main():
    '''
    Main function for orchestrating the naive softmax operation
    '''
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype = torch.float16)

    print(f'Naive softmax value for small values tensor: {naive_softmax(t1)}')
    # Highlighting numerical overflow error that occurs in naive softmax
    t2 = torch.tensor([[10.0, 11.0, 12.0], [8.0, 12.0, 13.0]], dtype = torch.float16)
    print('--- Numeric Overflow Example ---')
    print(f'Naive softmax value for risky tensor: {naive_softmax(t2)}')
        


if __name__ == '__main__':
    main()