'''
This file illustrates tiled flash attention on a 2-D tensor 
with batch_size = 1, seq_len = 4, d_k = 2 (attention head embedding dim), and n_heads = 1
for depicting the process in concrete step-by-step manner while deliberately avoiding 4-D complexity
'''

# Importing necessary libraries
import math
import torch
from config.settings import DEVICE

def tiled_flash_attention(Q: torch.tensor, K: torch.tensor, V: torch.tensor, qkv_block_size: int) -> torch.tensor:
    '''
    This function depicts tiled flash attention that directly addresses the seq_len ^ 2
    computational complexity by tiling Q, K and V and having only tiled dot product, and
    online softmax that greatly reduces the VRAM usage blowup that happens with increase in the
    sequence length.

    Args:
        Q: torch.tensor -> Query
        K: torch.tensor -> Key
        V: torch.tensor -> Value
        qkv_block_size: int -> No of tokens in a tile
    
    Returns:
        torch.tensor -> Attention output tensor  
    '''
    # Moving all the tensors to DEVICE
    Q, K, V = Q.to(DEVICE), K.to(DEVICE), V.to(DEVICE)
    # Here block size tiling happens along the sequence length i.e. each tile has Q, and K for block_size no of tokens
    # Getting the sequence length, and head dim from input Q
    # Q -> (seq_len, d_k)
    seq_len, d_k = Q.size() 
    # Ensuring seq_len is exactly divisible by block_size
    assert seq_len % qkv_block_size == 0, "Invalid block_size parameter specified"
    # Getting no of blocks
    n_blocks = seq_len // qkv_block_size
    # Reshaping Q, K and V from (seq_len, d_k) to (n_blocks, qkv_block_size, d_k)
    # Now each tensor will have n_blocks tiles each of shape (qkv_block_size, d_k)
    Q, K, V = Q.reshape(n_blocks, qkv_block_size, d_k), K.reshape(n_blocks, qkv_block_size, d_k), V.reshape(n_blocks, qkv_block_size, d_k)
    # Initializing tiled attention output tensor with 0s and is of shape (n_blocks, qkv_block_size, d_k)
    attn_output_tensor = torch.zeros((n_blocks, qkv_block_size, d_k))
    # In attention operation each query token is mutiplied with every key token, and their result with value token
    for i, q_block in enumerate(Q):
        # Online Softmax computation
        # max_so_far is initialized with -inf, and is of shape (qkv_block_size, 1)
        # Here we explicitly mention Q.dtype i.e. float16 as dtype as torch.full creates fp32 tensors by default
        max_so_far = torch.full((Q.size(1), 1), -torch.inf, dtype = Q.dtype, device = DEVICE)
        # denom_so_far is initialized with 0, and is of shape (qkv_block_size, 1)
        denom_so_far = torch.full((Q.size(1), 1), 0, dtype = Q.dtype, device = DEVICE)
        # output_so_far is initialized with 0, and is of shape (qkv_block_size, 1)
        output_so_far = torch.full((Q.size(1), d_k), 0, dtype = Q.dtype, device = DEVICE)
        # Iterating through K, and V tiles for attention computation with each query tile
        for k_block, v_block in zip(K, V):
            # Computing tiled attention score
            # (Q.K_T) / sqrt(d_k)
            # (qkv_block, d_k) @ (d_k, qkv_block) -> (qkv_block, qkv_block)
            tiled_attn_score = q_block @ k_block.transpose(0, 1) / math.sqrt(d_k)
            # Maxima for the current tile
            local_max = torch.max(tiled_attn_score, dim = 1, keepdims = True).values
            # Updated new maxima that's the maximum of running maximum, and local maxima
            new_max = torch.maximum(max_so_far, local_max)
            # Local stable exponent term
            exp_term = torch.exp(tiled_attn_score - new_max)
            # Exponent normalization term
            exp_diff = torch.exp(max_so_far - new_max)
            # Denominator for the current tile
            local_denom = torch.sum(exp_term, dim = 1, keepdims = True)
            # softmax(Q.K_T / sqrt(d_k)) for the current tile
            # (qkv_block, qkv_block) @ (qkv_block, d_k) -> (qkv_block, d_k)
            local_output = exp_term @ v_block
            # Output normalized in accordance to new maxima using exponent diff term
            output_so_far = output_so_far * exp_diff + local_output
            # Denominator normalized in accordance to the new maxima using exponent diff term
            denom_so_far = denom_so_far * exp_diff + local_denom
            # Running maximum is updated to the new maximum value
            max_so_far = new_max

        # Overall softmax score i.e. attention weight (accumulated over each tile)
        # (n_blocks, qkv_block, d_k)
        attn_output_tensor[i] = output_so_far / denom_so_far

    # Final attention output tensor
    # (seq_len, d_k)
    attn_output_tensor = attn_output_tensor.reshape(n_blocks * qkv_block_size, d_k)

    return attn_output_tensor

def main() -> None:
    '''
    Orcehstrator function that calls the tiled flash attention function
    '''
    # Query, Key and Value tensor definition
    Q = torch.tensor([[1, 0], [0, 1], [1, 1], [2, 1]], dtype = torch.float16)
    K = torch.tensor([[1, 0], [0, 1], [1, 1], [2, 1]], dtype = torch.float16)
    V = torch.tensor([[10, 0], [0, 20], [30, 30], [40, 40]], dtype = torch.float16)
    qk_block_size = 2

    print(f'Tiled flash attention output: {tiled_flash_attention(Q, K, V, qk_block_size)}')


if __name__ == '__main__':
    main()





