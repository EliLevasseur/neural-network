# PERFORMANCE LOGS


## Performance Results by Machine

<details>
<summary><b>2026-10-29: Matrix Multiplication Optimization</b></summary>
<br>

  <!-- Machine A Info -->
  <details>
  <summary><b>Machine 1: MacOs</b></summary>

 ### Benchmark Setup

- CPU: Intel(R) Core(TM) i7-1068NG7 CPU @ 2.30GHz
- Compiler: Apple clang 21, `-std=c++17 -O2`
- Benchmark: `benchmarks/matmul_benchmark.cpp`
- Matrix sizes: 100×100 × 100×100
- Number of multiplications: 1000 (2 billion FLOPs in total)
- Before: `naiveMatmul` (i-j-k). After: `operator*` (i-k-j)

  ### Performance Metrics
  **3.1× faster** matrix multiplication.

  | Metric | Before | After |
  | :--- | :--- | :--- |
  | Latency | ***1035.68ms** | **333.286ms** |
  | Memory Allocations | **240240000** | ****240240000** |

  </details>

  <!-- Machine B Info -->
  <details>
  <summary><b>Machine 2: PC</b></summary>

  ### Performance Metrics
  - to be tested


  </details>

</details>