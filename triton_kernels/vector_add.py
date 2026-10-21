'''
This code depicts writing a triton kernel for adding 2 vectors
'''
# Importing necessary libraries
import torch
import triton
import triton.language as tl
from config.settings import DEVICE

# This decorator makes vec_add a triton kernel and not a normal function
@triton.jit
def vec_add(x_ptr, y_ptr, output_ptr, n, block_size: tl.constexpr):
    '''
    This function performs a simple arithmetic addition of 2 vectors
    using triton programs each accomodating a block_size elements 
    from the vectors in the operation. Also the kernel itself doesn't return anything,
    rather the python function that wraps this kernel should return the output

    Args:
        x_ptr: pointer to the first element of x
        y_ptr: pointer to the first element of y
        n: No of elements in the vectors
        block_size: tl.constexpr
    '''
    # Getting progam ID
    # axis refers to triton's launchgrid axis
    # Since the operation involves 1-D vectors, the axis is set to 0
    pid = tl.program_id(axis = 0)
    # Starting index of the program
    program_start = pid * block_size
    # Computing offset (range of blocks over which the program will execute)
    offset = program_start + tl.arange(0, block_size)
    # Creating the mask (Mask hides the invalid i.e. out of bounds blocks for the operation)
    # All blocks that go from 0 to n - 1 (last element) will be valid
    mask = offset < n
    # Loading x and y from their starting pointers
    x = tl.load(x_ptr + offset, mask = mask)
    y = tl.load(y_ptr + offset, mask = mask)
    # Computing the output(sum)
    output = x + y
    # Saving the output to the output location that spans 
    # from starting index pointed to by output_ptr to output_ptr + offset
    tl.store(output_ptr + offset, output, mask = mask)

def vec_add_wrapper(x: torch.tensor, y:torch.tensor, block_size: int) -> torch.tensor:
    '''
    This python function acts as wrapper for the vec_add triton kernel
    Args:
        x: torch.tensor -> Input tensor 1 
        y: torch.tensor -> Input tensor 2
    
    Returns:
        output: torch.tensor -> Sum of x and y
    '''
    # Getting the no of elements in the vector
    n = x.numel()
    # Asserting the dtypes and sizes as these incompatibilites can cause kernel level errors
    assert x.is_cuda and y.is_cuda
    # Ensuring both are of same dims
    assert x.size() == y.size()
    # Ensuring both have same dtype
    assert x.dtype == y.dtype
    # Ensuring both of these are stored in contiguous memory locations
    assert x.is_contiguous() and y.is_contiguous()
    # Checking if block_size is less than the total no of elements
    assert block_size <= n

    # Creating an output tensor
    output = torch.empty_like(x, dtype = torch.float32)
    # Creating a triton launchgrid
    grid = (triton.cdiv(n, block_size), )
    # Calling the triton kernel for this python wrapper
    vec_add[grid](x, y, output, n, block_size)

    return output


def main():
    '''
    This function runs the vector addition triton kernel
    '''
    x = torch.randn(100, dtype = torch.float32, device = DEVICE)
    y = torch.randn(100, dtype = torch.float32, device = DEVICE)

    direct_sum = x + y
    triton_kernel_sum = vec_add_wrapper(x, y, 16)

    print(f'Direct sum: {direct_sum}')
    print(f'Triton kernel sum: {triton_kernel_sum}')

    print(f'Is triton sum equal to direct sum ?: {torch.equal(direct_sum, triton_kernel_sum)}')

if __name__ == '__main__':
    main()

