'''
This file implements stable softmax operation on a 2-D Tensor, that mitigates numerical overflow error
'''
# Importing necessary libraries
import torch
from config.settings import DEVICE

def stable_softmax(x: torch.tensor) -> torch.tensor:
    '''
    This function implements stable softmax (with global maxima diff in exponent)
    on a 2-D tensor. The only change in the stable softmax operation is that 
    every exponent is e ^ x_i - max_element in that row. This ensures that 
    numeric values never blow past the range that can be supported.

    Args:
        x: torch.tensor -> Input tensor
    
    Returns:
        torch.tensor -> Stable softmax (Input tensor)
    '''
    # Finding the row-wise maxima
    # Here keepdims ensures row_max is (2, 1) i.e. 1 max value per row
    # Moving input tensor to device
    x = x.to(DEVICE)
    row_max = torch.max(x, dim = 1, keepdims = True)[0]

    # Creates a tensor of every value of X raised to exp - maxima for that row
    num = torch.exp(x - row_max) 
    # Performs a row-wise sum of exp(X) - maxima for that row
    # keepdims ensures that denom tensor is 2-D (2, 1) which is required for tensor broadcasting
    denom = torch.sum(num , dim = -1, keepdims = True)
    stable_sm = num / denom

    return stable_sm

def main() -> None:
    '''
    Main function for orchestrating the stable softmax operation
    '''
    t1 = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype = torch.float16)

    print(f'Stable softmax value for small values tensor: {stable_softmax(t1)}')
    # Highlighting how stable softmax mitigates the numerical overflow avoided in naive softmax
    t2 = torch.tensor([[10.0, 11.0, 12.0], [8.0, 12.0, 13.0]], dtype = torch.float16)
    print('--- Numeric Overflow Mitigated ---')
    print(f'Naive softmax value for risky tensor: {stable_softmax(t2)}')

if __name__ == '__main__':
    main()