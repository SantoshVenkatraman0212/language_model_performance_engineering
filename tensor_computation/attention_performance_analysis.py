'''
This code performs a comparative analysis of naive and tiled flash attention 
considering criteria such as: range specific numeric equivalence, error / diff,
execution time, peak memory consumption and precision
'''
# Importing necessary libraries
import matplotlib.pyplot as plt
import time
import torch
from config.settings import DEVICE
from .naive_attention import naive_attention_comp
from .tiled_flash_attention import tiled_flash_attention_comp
from config.paths import ATTN_COMPARISON_IMG_DIR

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

def main():
    '''
    Orchestrator function for running the attention benchmarks
    '''
    # 1. Precision (FP32 VS FP16)
    # FP-16
    Q1 = torch.tensor([[1, 0], [0, 1], [1, 1], [2, 1]], dtype = torch.float32)
    K1 = torch.tensor([[1, 0], [0, 1], [1, 1], [2, 1]], dtype = torch.float32)
    V1 = torch.tensor([[10, 0], [0, 20], [30, 30], [40, 40]], dtype = torch.float32)
    print('---------- 1. Precision (FP32 VS FP16) ----------')
    print('----- FP32 -----')
    # Warming up the GPU (CUDA kernel initialization itself will consume lots of time)
    # This will severely increase the time taken for the first attention to run
    # 10 iterations for warmup
    for i in range(10):
        print(f'Warmup phase - {i + 1}')
        naive_attention_comp(Q1, K1, V1)
        tiled_flash_attention_comp(Q1, K1, V1, 2)
        
    torch.cuda.synchronize()
    start = time.time()
    naive_attn_output = naive_attention_comp(Q1, K1, V1)
    torch.cuda.synchronize()
    end = time.time()
    naive_attn_time = end - start
    torch.cuda.synchronize()
    start = time.time()
    flash_attn_output = tiled_flash_attention_comp(Q1, K1, V1, 2)
    torch.cuda.synchronize()
    end = time.time()
    flash_attn_time = end - start
    print(f'Naive Attention Output\n{naive_attn_output}')
    print(f'Flash Attention Output\n{flash_attn_output}')
    num_corr_dict = numeric_correctness(naive_attn_output, flash_attn_output)
    print('----- FP16 -----')
    Q1 = Q1.to(torch.float16)
    K1 = K1.to(torch.float16)
    V1 = V1.to(torch.float16)
    naive_attn_output = naive_attention_comp(Q1, K1, V1)
    flash_attn_output = tiled_flash_attention_comp(Q1, K1, V1, 2)
    print(f'Naive Attention Output\n{naive_attn_output}')
    print(f'Flash Attention Output\n{flash_attn_output}')
    # 2. Numeric Correctness (FP16)
    print('\n---------- 2. Numeric Correctness ----------')
    print(f'----- Numeric Equivalence -----\n{num_corr_dict['numeric_equivalence']}')
    print(f'----- Absolute Error -----\n{num_corr_dict['absolute_error']}')
    print(f'----- Mean Absolute Error -----\n{num_corr_dict['mean_absolute_error']}')
    print(f'----- Max Absolute Error -----\n{num_corr_dict['max_absolute_error']}')
    # 3. Execution time (FP16)
    print('\n---------- 3. Execution Time ----------')
    print(f'----- Naive attention -----\n{naive_attn_time}s')
    print(f'----- Tiled flash attention time -----\n{flash_attn_time}s')
    # 4. Block size peak VRAM usage comparison (tiled flash attention)
    # For the 2-D Input tensors with sequence length 4, the block sizes can be 1, 2 and 4
    # block size = 1
    torch.cuda.reset_peak_memory_stats()
    o1 = tiled_flash_attention_comp(Q1, K1, V1, 1)
    peak_memory1 = torch.cuda.max_memory_allocated()
    # block size = 2
    torch.cuda.reset_peak_memory_stats()
    o2 = tiled_flash_attention_comp(Q1, K1, V1, 2)
    peak_memory2 = torch.cuda.max_memory_allocated()
    # block size = 4
    torch.cuda.reset_peak_memory_stats()
    o3 = tiled_flash_attention_comp(Q1, K1, V1, 4)
    peak_memory3 = torch.cuda.max_memory_allocated()
    print('\n---------- 4. Block size VS Max VRAM allocated ----------')
    print(f'------ Block size = 1 -----\nMax VRAM allocated: {peak_memory1 // (1024 ** 2)}MB')
    print(f'------ Block size = 2 -----\nMax VRAM allocated: {peak_memory2 // (1024 ** 2)}MB')
    print(f'------ Block size = 4 -----\nMax VRAM allocated: {peak_memory3 // (1024 ** 2)}MB')
    # 5. Peak VRAM usage for different sequence lengths
    # Creating random tensors for measuring scaling at different sequence lengths
    seq_len = [256, 512, 1024, 2048, 4096, 8192]
    naive_attn_times, flash_attn_times, naive_attn_memory, flash_attn_memory = [], [], [], []
    print('\n---------- Naive VS Flash attention scaling with sequence length ----------')
    # Setting seed for reproducibility
    torch.random.manual_seed(42)
    for i, s in enumerate(seq_len):
        Q = torch.randn((s, 32), dtype = torch.float16)
        K = torch.randn((s, 32), dtype = torch.float16)
        V = torch.randn((s, 32), dtype = torch.float16)
        
        # Naive attention
        torch.cuda.synchronize()
        start = time.time()
        torch.cuda.reset_peak_memory_stats()
        attn_output = naive_attention_comp(Q, K, V)
        torch.cuda.synchronize()
        end = time.time()
        time_diff = end - start
        naive_attn_times.append(time_diff)
        attn_peak_memory = (torch.cuda.max_memory_allocated()) // (1024 ** 2)
        naive_attn_memory.append(attn_peak_memory)
        # Flash attention
        torch.cuda.synchronize()
        start = time.time()
        torch.cuda.reset_peak_memory_stats()
        attn_output = tiled_flash_attention_comp(Q, K, V, 32)
        torch.cuda.synchronize()
        end = time.time()
        time_diff = end - start
        flash_attn_times.append(time_diff)
        attn_peak_memory = torch.cuda.max_memory_allocated() // (1024 ** 2)
        flash_attn_memory.append(attn_peak_memory)

        print(f'Experiment - {i + 1}')
        print(f'Sequence length = {s}')
        print(f'----- Naive Attention -----\nExecution time: {naive_attn_times[i]}s\nPeak VRAM allocated: {naive_attn_memory[i]}MB')
        print(f'----- Flash Attention -----\nExecution time: {flash_attn_times[i]}s\nPeak VRAM allocated: {flash_attn_memory[i]}MB')

    # Plotting how peak memory allocation, and execution time varies for flash and naive attention with varying sequence lengths
    # Sequence length vs peak VRAM allocation
    plt.figure(figsize = (20, 10))
    plt.subplot(1, 2, 1)
    plt.plot(seq_len, naive_attn_memory, c = 'r', label = 'Naive attention peak VRAM')
    plt.plot(seq_len, flash_attn_memory, c = 'b', label = 'Flash attention peak VRAM')
    plt.title('Peak VRAM allocated for naive & flash attention VS sequence length')
    plt.xlabel('Sequence length (No of tokens in a sequence)')
    plt.ylabel('Peak VRAM allocation for naive (red) & flash (blue) attention (in MB)')
    plt.legend()
    plt.tight_layout()
    plt.subplot(1, 2, 2)
    plt.plot(seq_len, naive_attn_times, c = 'r', label = 'Naive attention execution time')
    plt.plot(seq_len, flash_attn_times, c = 'b', label = 'Flash attention execution time')
    plt.title('Execution for naive & flash attention VS sequence length')
    plt.xlabel('Sequence length (No of tokens in a sequence)')
    plt.ylabel('Execution time for naive (red) & flash (blue) attention (in secs)')
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'{ATTN_COMPARISON_IMG_DIR}/Attention_sequence_length_scaling.png', dpi = 300, bbox_inches = 'tight')

if __name__ == '__main__':
    main()

    

