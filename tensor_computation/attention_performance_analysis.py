'''
This code performs a comparative analysis of naive and tiled flash attention 
considering criteria such as: range specific numeric equivalence, error / diff,
execution time, peak memory consumption and precision
'''
# Importing necessary libraries
import math
import torch
from time import time
from config.settings import DEVICE
from .naive_attention import naive_attention_comp
from .tiled_flash_attention import tiled_flash_attention_comp

def numeric_correctness(attn_o1: torch.tensor, attn_o2: torch.tensor):
    '''
    This function compares the
    '''
    # Getting the numerical equivalence of naive and tiled flash attention outputs using torch.allclose()
    num_eq = torch.allclose(attn_o1, attn_o2)
    # Absolute error
    abs_err = abs(attn_o1 - attn_o2)
    # Mean absolute error
    mean_abs_err = torch.mean(abs_err)
    # Max absolute error
    max_abs_err = torch.max(abs_err)

    return {'numeric_equivalence': num_eq, 'absolute_error': abs_err, 'mean_absolute_error': mean_abs_err, 'max_absolute_error': max_abs_err}

def attention_benchmark(Q: torch.tensor, K: torch.tensor, V: torch.tensor, qkv_block_size: int):
    '''
    
    '''
    return {'naive_attention_output': naive_attention_comp(Q, K, V), 'tiled_flash_attention_output': tiled_flash_attention_comp(Q, K, V, qkv_block_size)}

def main():
    '''
    Orchestrator function for running the attention benchmarks
    '''
    # 1. Precision (FP32 VS FP16)
    # FP-16
    Q1 = torch.tensor([[1, 0], [0, 1], [1, 1], [2, 1]], dtype = torch.float16)
    K1 = torch.tensor([[1, 0], [0, 1], [1, 1], [2, 1]], dtype = torch.float16)
    V1 = torch.tensor([[10, 0], [0, 20], [30, 30], [40, 40]], dtype = torch.float16)
    print('---------- 1. Precision (FP32 VS FP16) ----------')
    print('----- FP16 -----')
    attn_output_dict = attention_benchmark(Q1, K1, V1, 2)
    print(f'Naive Attention Output\n{attn_output_dict['naive_attention_output']}')
    print(f'Flash Attention Output\n{attn_output_dict['tiled_flash_attention_output']}')
    print('----- FP32 -----')
    Q2 = torch.tensor([[1, 0], [0, 1], [1, 1], [2, 1]], dtype = torch.float32)
    K2 = torch.tensor([[1, 0], [0, 1], [1, 1], [2, 1]], dtype = torch.float32)
    V2 = torch.tensor([[10, 0], [0, 20], [30, 30], [40, 40]], dtype = torch.float32)
    attn_output_dict = attention_benchmark(Q2, K2, V2, 2)
    print(f'Naive Attention Output\n{attn_output_dict['naive_attention_output']}')
    print(f'Flash Attention Output\n{attn_output_dict['tiled_flash_attention_output']}')
    # 2. Numeric Correctness (FP16)
    print('\n---------- 2. Numeric Correctness ----------')
    num_corr_dict = numeric_correctness(attn_output_dict['naive_attention_output'], attn_output_dict['tiled_flash_attention_output'])
    print(f'----- Numeric Equivalence -----\n{num_corr_dict['numeric_equivalence']}')
    print(f'----- Absolute Error -----\n{num_corr_dict['absolute_error']}')
    print(f'----- Mean Absolute Error -----\n{num_corr_dict['mean_absolute_error']}')
    print(f'----- Max Absolute Error -----\n{num_corr_dict['max_absolute_error']}')

if __name__ == '__main__':
    main()

    

