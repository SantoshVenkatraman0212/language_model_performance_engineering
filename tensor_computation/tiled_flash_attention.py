'''
This file illustrates tiled flash attention on a 2-D tensor 
with batch_size = 1, seq_len = 4, d_k = 2 (attention head embedding dim), and n_heads = 1
for depicting the process in concrete step-by-step manner while deliberately avoiding 4-D complexity
'''

# Importing necessary libraries
import math
import torch

def tiled_flash_attention(Q: torch.tensor, K: torch.tensor, V: torch.tensor, qk_block_size: int) -> torch.tensor:
    # Here block size tiling happens along the sequence length i.e. each tile has Q, and K for block_size no of tokens
    # Getting the sequence length, and head dim from input Q
    seq_len, d_k = Q.size()
    # Ensuring seq_len is exactly divisible by block_size
    assert seq_len % qk_block_size == 0, "Invalid block_size parameter specified"
    # Getting no of blocks
    n_blocks = seq_len // qk_block_size
    # Reshaping Q, and K accordingly
    Q, K, V = Q.reshape(n_blocks, qk_block_size, d_k), K.reshape(n_blocks, qk_block_size, d_k), V.reshape(n_blocks, qk_block_size, d_k)
    attn_output_tensor = torch.zeros((n_blocks, qk_block_size, d_k))
    for i, q_block in enumerate(Q):
        # Online Softmax computation
        max_so_far = torch.full((Q.size(1), 1), -torch.inf, dtype = Q.dtype)
        denom_so_far = torch.full((Q.size(1), 1), 0, dtype = Q.dtype)
        output_so_far = torch.full((Q.size(1), d_k), 0, dtype = Q.dtype)
        for k_block, v_block in zip(K, V):
            # Computing tiled attention score
            tiled_attn_score = q_block @ k_block.transpose(0, 1) / math.sqrt(d_k)

            local_max = torch.max(tiled_attn_score, dim = 1, keepdims = True).values
            new_max = torch.maximum(max_so_far, local_max)
            exp_term = torch.exp(tiled_attn_score - new_max)
            exp_diff = torch.exp(max_so_far - new_max)
            local_denom = torch.sum(exp_term, dim = 1, keepdims = True)
            local_output = exp_term @ v_block
            output_so_far = output_so_far * exp_diff + local_output
            denom_so_far = denom_so_far * exp_diff + local_denom
            max_so_far = new_max

        # Overall softmax score i.e. attention weight
        attn_output_tensor[i] = output_so_far / denom_so_far

    attn_output_tensor = attn_output_tensor.reshape(n_blocks * qk_block_size, d_k)

    return attn_output_tensor

def main():
    # Query, Key and Value tensor definition
    Q = torch.tensor([[1, 0], [0, 1], [1, 1], [2, 1]], dtype = torch.float16)
    K = torch.tensor([[1, 0], [0, 1], [1, 1], [2, 1]], dtype = torch.float16)
    V = torch.tensor([[10, 0], [0, 20], [30, 30], [40, 40]], dtype = torch.float16)
    qk_block_size = 2

    print(f'Tiled flash attention output: {tiled_flash_attention(Q, K, V, qk_block_size)}')


if __name__ == '__main__':
    main()





