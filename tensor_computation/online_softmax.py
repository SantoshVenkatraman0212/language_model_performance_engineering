'''
This file depicts online tiled softmax that enables modern language model
pipelines to implement numerically stable softmax without having to have the 
entire sequence in memory at once (seq_len scales up as a squared term for attention computation)
'''

# Importing necessary libraries
import torch
from config.settings import DEVICE

def online_tiled_softmax(x: torch.tensor, block_size: int) -> torch.tensor:
    '''
    This function implements online tiled softmax, where the input tensor is first
    reshaped (divided) into multiple tiles of specified compatible block size. Then a running 
    per-row max tensor, and denominator tensors are maintained. This is essential for scaling of 
    numerator and denominator according to the new maxima

    Args:
        x: torch.tensor -> Input tensor
        block_size: int -> desired block size for each input tensor tile
    
    Returns:
        torch.tensor -> Online tiled softmax (Input tensor)
    '''
    # Moving the input tensor to device
    x = x.to(DEVICE)
    # Here rows are batch_size, and cols are sequence lengths
    rows, cols = x.size()
    # Ensure that the no of cols is exactly divisible by block_size for tiling to be possible
    assert cols % block_size == 0, 'Invalid block_size for online softmax'
    # No of blocks in each input tensor tile
    n_blocks = cols // block_size
    # The tiled tensor is of shape (n_blocks, rows, block_size) after reshape
    # So there are n_blocks tiles each of shape (rows, block_size)
    tiled_tensor = x.reshape(rows, n_blocks, block_size).transpose(0, 1)
    # max_so_far tensor is initialized with -inf and is of shape (n_blocks, 1)
    max_so_far = torch.full((x.shape[0], 1), -torch.inf, device = DEVICE)
    # denom_so_far is initialized as 0 and is of shape (n_blocks, 1)
    denom_so_far = torch.zeros((x.shape[0], 1), device = DEVICE)
    # Since the incremental softmax computation per tile is sequential, a loop is required
    # This is contrary to the always preferred parallel tensor operations but the main goal
    # here is to avoid materializing the entire sequence tensors in memory simultaneously
    # Iterating through each tile
    for block in tiled_tensor:
        # local maxima for that tile
        max_local = torch.max(block, dim = 1, keepdims = True).values
        # Updated maxima is max of whatever maximum value was so far and local maxima
        max_new = torch.maximum(max_so_far, max_local)
        # e ^ max_so_far - new maxima is the normalizing exponent term
        # This is required so that all the tile-specific softmax outputs are normalized to 
        # new maximum values per row
        exp_diff = torch.exp(max_so_far - max_new)
        # stable softmax numerator for the current tile
        exp_num = torch.exp(block - max_new)
        # stable softmax denominator for the current tile
        local_denom = torch.sum(exp_num, dim = 1, keepdims = True)
        # The denominator in tiled online softmax should be normalized and accumulated across each tile
        # Therefore, the old denominator is normalized with exponential difference, and added to the 
        # local denominator of the current tile 
        denom_so_far = denom_so_far * exp_diff + local_denom
        # The new maxima becomes the overall (so-far) maxima
        max_so_far = max_new

    # Now that we have overall normalized denominator, and the final max tensor
    # softmax can be directly computed
    softmax_value = torch.exp(x - max_so_far) / denom_so_far

    return softmax_value

def main():
    '''
    Orchestrator function for the online tiled softmax operation
    '''
    t1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype = torch.float16)
    print(f'Tiled softmax output: {online_tiled_softmax(t1, 2)}')

if __name__ == '__main__':
    main()

    



