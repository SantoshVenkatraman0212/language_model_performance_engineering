'''
This file performs naive attention on 2-D input tensors (Q, K and V) by materializing the entire sequence 
at once in the memory without tiled computation and online softmax 
'''
# Importing necessary libraries
import torch
import math
from config.settings import DEVICE

def naive_attention_comp(Q: torch.tensor, K: torch.tensor, V: torch.tensor) -> torch.tensor:
    '''
    This function implements naive attention on the input 2-D Q, K and V tensors without tiling
    and online softmax, i.e. entire sequences are processed at once in the compute device 
    rather than sequential processing of tiles (attention), and naive stable softmax (direct
    global maxima, and denominator)

    Args:
        Q: torch.tensor ->  Query
        K: torch.tensor -> Key
        V: torch.tensor -> Value
    
    Returns:
        attn_out: torch.tensor -> Output from attention block
    '''
    # Moving all the tensors to the compute device
    Q, K, V = Q.to(DEVICE), K.to(DEVICE), V.to(DEVICE)
    # Here we're only taking 2-D tensors and not 4-D, so the shape is (seq_len, d_k) 
    # This means here, batch_size = 1, n_heads = 1
    # Getting the sequence length and attention head embedding dim from Q
    seq_len, d_k = Q.size()
    # Computing attention score using (Q.K_transpose) / sqrt(d_k)
    # Q.size() -> (seq_len, d_k) | K.size() -> (seq_len, d_k)
    # Therefore taking transpose of K, attn_sc.size() -> (seq_len, d_k) @ (d_k, seq_len) = (seq_len, seq_len)
    attn_sc = Q @ K.transpose(0, 1) / math.sqrt(d_k)
    # Numerically stable softmax computation
    # Row-wise max tensor
    max_tensor = torch.max(attn_sc, dim = 1, keepdim = True).values
    # Numerator tensor (e^x - max for that row)
    num = torch.exp(attn_sc - max_tensor)
    # Denominator tensor sum(e^x - max for that row) for the entire row
    denom = torch.sum(num, dim = 1, keepdim = True)
    # Attention score (softmax(Q.K_transpose) / sqrt(d_k))
    attn_sc = num / denom
    # Computing attention output (attention_score @ Value tensor)
    # (seq_len, seq_len) @ (seq_len, d_k) -> (seq_len, d_k) therefore, direct product without transpose of V is performed here
    attn_out = attn_sc @ V

    return attn_out

def main():
    '''
    Orchestrator function for the naive attention function
    '''
    Q = torch.tensor([[1, 0], [0, 1], [1, 1], [2, 1]], dtype = torch.float16)
    K = torch.tensor([[1, 0], [0, 1], [1, 1], [2, 1]], dtype = torch.float16)
    V = torch.tensor([[10, 0], [0, 20], [30, 30], [40, 40]], dtype = torch.float16)

    print(f'Naive attention output: {naive_attention_comp(Q, K, V)}')


if __name__ == '__main__':
    main()