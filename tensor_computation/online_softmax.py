# Importing necessary libraries
import torch

def online_tiled_softmax(x: torch.tensor, block_size: int) -> torch.tensor:
    '''
    A simplified demo of online softmax on a 2-D tensor
    '''
    # Here rows are batch_size, and cols are sequence lengths
    rows, cols = x.size()
    # Ensure that the no of cols is exactly divisible by block_size for tiling to be possible
    assert cols % block_size == 0, 'Invalid block_size for online softmax'
    n_blocks = cols // block_size

    tiled_tensor = x.reshape(rows, n_blocks, block_size).transpose(0, 1)
    max_so_far = torch.full((x.shape[0], 1), -torch.inf)
    denom_so_far = torch.zeros((x.shape[0], 1))

    for block in tiled_tensor:
        # local maxima
        max_local = torch.max(block, dim = 1, keepdims = True).values
        max_new = torch.maximum(max_so_far, max_local)
        exp_diff = torch.exp(max_so_far - max_new)
        exp_num = torch.exp(block - max_new)
        local_denom = torch.sum(exp_num, dim = 1, keepdims = True)
        denom_so_far = denom_so_far * exp_diff + local_denom
        max_so_far = max_new

    softmax_value = torch.exp(x - max_so_far) / denom_so_far

    return softmax_value

def main():
    t1 = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype = torch.float16)
    print(f'Tiled softmax output: {online_tiled_softmax(t1, 2)}')

if __name__ == '__main__':
    main()

    



