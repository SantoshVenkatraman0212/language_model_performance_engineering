'''
This file implements numerically stable softmax using triton fused kernels for
2-D tensor
'''
# Importing necessary libraries
import torch
import triton
import triton.language as tl
from config.settings import DEVICE
from tensor_computation.stable_softmax import stable_softmax

@triton.jit
def fused_softmax_kernel(input_ptr, output_ptr, row_stride: int, col_stride: int, 
                         ncols: int, block_size: tl.constexpr):
    '''
    This triton kernel function implements numerically stable softmax
    on a 2-D tensor using fused kernel. i.e. each program loads an entire
    row from DRAM into the VRAM computes, max of that row, computes e^x - max
    calculates the denominator and stores that result.

    Args:
        input_ptr: points to the address of (0, 0)th element
        output_ptr: points to the starting address where the output should be saved
        row_stride: PyTorch stride by which that specific row is shifted in the memory
        col_stride: PyTorch stride by which that specific col is shifted in the memory
        ncols: int -> Total no of cols in the 2-D input tensor
        block_size: constexpr -> block size refers to the no of elements involved in one program's operation
        It's constexpr type that tells triton that this value should be materialized only during compute
        & needn't be in the memory all the time
    '''
    # Getting the program ID
    # Even for a 2-D tensor, since each program is handling a 1-D vector, axis is 0
    pid = tl.program_id(axis = 0)
    # Computing offset
    # For this we need use row and col strides to get the memory locations of each value in the 2-D tensor
    # as they needn't always be stored in contiguous memory locations
    # Here the logic is in every program an entire row is loaded into the VRAM for computation
    # pid is the program ID (0, 1, 2, ....) & pid * row_stride gives the location of the first element in
    # the current program's row
    # We need a separate col index vector for non-contiguous case
    col_index = tl.arange(0, block_size)
    # tl.arange * col_stride gives the range of addresses of all values in that operation
    offset = pid * row_stride + col_index * col_stride
    # Mask ensures invalid memory addresses (Out of Bounds) aren't accessed
    mask = col_index < ncols 
    # Loading the entire row vector with mask
    row = tl.load(input_ptr + offset, mask = mask, other = -torch.inf)
    # Max for that row
    m = tl.max(row, axis = 0)
    # Numerator
    num = tl.exp(row - m)
    # Denominator
    denom = tl.sum(num, axis = 0)
    # Softmax (numerically stable as exp(x - max) avoids numeric overflow)
    output = num / denom
    # Storing the output of softmax of that particular row
    tl.store(output_ptr + offset, output, mask = mask)

def fused_softmax_wrapper(x: torch.tensor) -> torch.tensor:
    '''
    Python wrapper for the fused_softmax_kernel function
    Args:
        x: torch.tensor -> 2-D Input tensor
    Returns:
        output: torch.tensor -> softmax(x)
    '''
    # Ensuring x is in GPU
    assert x.is_cuda
    # Getting the row, col and stride from x
    row, col = x.size()
    stride = x.stride()
    # Here the block_size is taken to be exactly equal to the no of cols for simplicity to load the entire
    # row in one program
    block_size = col
    # Creating an empty output tensor
    output = torch.empty_like(x, dtype = torch.float32, device = DEVICE)
    # Creating a 2-D triton launchgrid
    grid = (row,)
    # Calling the kernel
    fused_softmax_kernel[grid](x, output, stride[0], stride[1], col, block_size)

    return output

def main():
    '''
    Function that runs the fused stable softmax and compares the result with direct PyTorch implementation
    '''
    t1 = torch.randn((2, 4), dtype = torch.float32, device = DEVICE) 
    fused_softmax = fused_softmax_wrapper(t1)
    # PyTorch softmax
    torch_softmax = stable_softmax(t1)
    print(f'----- Triton fused kernel softmax -----\n{fused_softmax}')
    print(f'----- PyTorch softmax -----\n{torch_softmax}')
    # The computation results produced by PyTorch and Triton can look identical but the bits used can be different
    # That's why the they aren't identical and this returns False
    print(f'Are the results same ?: {torch.equal(fused_softmax, torch_softmax)}')
    # Checking if the results lie within acceptable numberical difference range
    # This returns True
    print(f'Are the results similar within torch acceptable range ?: {torch.allclose(fused_softmax, torch_softmax)}')
if __name__ == '__main__':
    main()

    