import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the CSV file
csv_path = './Part_3/stats.csv'
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found. Please run part3_build_csv.py first.")
    exit(1)

# Load data
df = pd.read_csv(csv_path, names=['executable', 'N', 'M', 'P', 'time'])

# Filter out errors and timeouts
df = df[df['time'].apply(lambda x: isinstance(x, (int, float)) or (isinstance(x, str) and x.replace('.', '').replace('e', '').replace('-', '').replace('+', '').isdigit()))]
df['time'] = pd.to_numeric(df['time'], errors='coerce')
df = df.dropna(subset=['time'])

# Calculate mean time for each configuration
df_mean = df.groupby(['executable', 'N', 'M', 'P'])['time'].mean().reset_index()

# Create output directory for plots
os.makedirs('./Part_3/plots', exist_ok=True)

print("Generating performance analysis plots...")

# ========== Plot 1: Performance comparison for square matrices (N=M=P) ==========
df_square = df_mean[df_mean['N'] == df_mean['M']]
df_square = df_square[df_square['N'] == df_square['P']]

plt.figure(figsize=(12, 7))
for exec_name in df_square['executable'].unique():
    data = df_square[df_square['executable'] == exec_name]
    plt.plot(data['N'], data['time'], marker='o', label=exec_name, linewidth=2)

plt.xlabel('Matrix Dimension (N=M=P)', fontsize=12)
plt.ylabel('Time (seconds)', fontsize=12)
plt.title('Performance Comparison - Square Matrices', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.tight_layout()
plt.savefig('./Part_3/plots/performance_square_matrices.png', dpi=300)
print("  Saved: performance_square_matrices.png")
plt.close()

# ========== Plot 2: Speedup comparison relative to sequential ==========
df_square_pivot = df_square.pivot(index='N', columns='executable', values='time')
if 'matrix_mult_sequential' in df_square_pivot.columns:
    sequential_times = df_square_pivot['matrix_mult_sequential']
    
    plt.figure(figsize=(12, 7))
    for col in df_square_pivot.columns:
        if col != 'matrix_mult_sequential':
            speedup = sequential_times / df_square_pivot[col]
            plt.plot(df_square_pivot.index, speedup, marker='o', label=col, linewidth=2)
    
    plt.xlabel('Matrix Dimension (N=M=P)', fontsize=12)
    plt.ylabel('Speedup (vs Sequential)', fontsize=12)
    plt.title('Speedup Comparison - Square Matrices', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axhline(y=1, color='r', linestyle='--', label='Sequential baseline')
    plt.tight_layout()
    plt.savefig('./Part_3/plots/speedup_square_matrices.png', dpi=300)
    print("  Saved: speedup_square_matrices.png")
    plt.close()

# ========== Plot 3: Performance comparison by precision (double vs float) ==========
precision_execs = ['matrix_mult_cuda_shared', 'matrix_mult_cuda_float']
df_precision = df_square[df_square['executable'].isin(precision_execs)]

if not df_precision.empty:
    plt.figure(figsize=(12, 7))
    for exec_name in precision_execs:
        data = df_precision[df_precision['executable'] == exec_name]
        if not data.empty:
            label = exec_name.replace('matrix_mult_cuda_', '').replace('_', ' ').title()
            plt.plot(data['N'], data['time'], marker='o', label=label, linewidth=2)
    
    plt.xlabel('Matrix Dimension (N=M=P)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Precision Comparison (Double vs Float)', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('./Part_3/plots/precision_comparison.png', dpi=300)
    print("  Saved: precision_comparison.png")
    plt.close()

# ========== Plot 4: 1 Thread vs Shared Memory Performance ==========
compare_execs = ['matrix_mult_cuda_1thread', 'matrix_mult_cuda_shared']
df_compare = df_square[df_square['executable'].isin(compare_execs)]

if not df_compare.empty:
    plt.figure(figsize=(12, 7))
    for exec_name in compare_execs:
        data = df_compare[df_compare['executable'] == exec_name]
        if not data.empty:
            label = '1 Thread per Block' if '1thread' in exec_name else 'Shared Memory (16x16)'
            plt.plot(data['N'], data['time'], marker='o', label=label, linewidth=2, markersize=8)
    
    plt.xlabel('Matrix Dimension (N=M=P)', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('CUDA Optimization: 1 Thread vs Shared Memory', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('./Part_3/plots/cuda_optimization_comparison.png', dpi=300)
    print("  Saved: cuda_optimization_comparison.png")
    plt.close()

# ========== Plot 5: GFLOPS comparison ==========
# Calculate GFLOPS: (2 * N * M * P) / (time * 1e9)
df_square['gflops'] = (2 * df_square['N'] * df_square['M'] * df_square['P']) / (df_square['time'] * 1e9)

plt.figure(figsize=(12, 7))
for exec_name in df_square['executable'].unique():
    data = df_square[df_square['executable'] == exec_name]
    plt.plot(data['N'], data['gflops'], marker='o', label=exec_name, linewidth=2)

plt.xlabel('Matrix Dimension (N=M=P)', fontsize=12)
plt.ylabel('Performance (GFLOPS)', fontsize=12)
plt.title('Performance in GFLOPS - Square Matrices', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('./Part_3/plots/gflops_comparison.png', dpi=300)
print("  Saved: gflops_comparison.png")
plt.close()

# ========== Generate statistics table ==========
print("\n" + "="*80)
print("PERFORMANCE STATISTICS - Square Matrices (N=M=P)")
print("="*80)

for dim in sorted(df_square['N'].unique()):
    print(f"\nMatrix Dimension: {dim} x {dim} x {dim}")
    print("-" * 80)
    dim_data = df_square[df_square['N'] == dim].sort_values('time')
    
    for _, row in dim_data.iterrows():
        exec_name = row['executable'].replace('matrix_mult_', '').replace('_', ' ').title()
        time = row['time']
        gflops = row['gflops']
        
        if 'matrix_mult_sequential' in df_square['executable'].values:
            seq_time = df_square[(df_square['N'] == dim) & (df_square['executable'] == 'matrix_mult_sequential')]['time'].values
            if len(seq_time) > 0:
                speedup = seq_time[0] / time
                print(f"  {exec_name:40s}: {time:8.4f}s  {gflops:8.2f} GFLOPS  Speedup: {speedup:6.2f}x")
            else:
                print(f"  {exec_name:40s}: {time:8.4f}s  {gflops:8.2f} GFLOPS")
        else:
            print(f"  {exec_name:40s}: {time:8.4f}s  {gflops:8.2f} GFLOPS")

print("\n" + "="*80)
print("All plots saved to ./Part_3/plots/")
print("="*80)
