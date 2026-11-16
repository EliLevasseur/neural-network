# PERFORMANCE LOGS


## Performance Results by Machine

<details>
<summary><b>2026-11-15: Matrix Multiplication Optimization</b></summary>
<br>

  <!-- Machine A Info -->
  <details>
  <summary><b>Machine 1: PC</b></summary>

 ### Benchmark Setup

- Optimization: loop order changed from i-j-k to i-k-j
- CPU: AMD Ryzen 9 9950X, Fedora Linux 44 on WSL
- Compiler: GCC 16.2.1
- Flags: `-std=c++17 -O3 -march=native -DNDEBUG`
- Benchmark: `benchmarks/matmul_benchmark.cpp`
- Matrix sizes: [128, 512] × [512, 128], the forward multiply of GLM's first Dense layer at batch 128
- Number of multiplications: 1000 (16.8 billion FLOPs in total)
- Before: `naiveMatmul` (i-j-k). After: `operator*` (i-k-j)
- Each number is the middle result of 3 runs

  ### Performance Metrics
  **9.0× faster** matrix multiplication.

  | Metric | Before | After |
  | :--- | :--- | :--- |
  | Time | **5039.69 ms** | **558.96 ms** |
  | GFLOP/s | **3.33** | **30.0** |
  | Memory Allocated | **131152000 bytes** | **131152000 bytes** |

  </details>

  <!-- Machine B Info -->
  <details>
  <summary><b>Machine 2: MacOs</b></summary>

  ### Performance Metrics
  - to be tested


  </details>

</details>
