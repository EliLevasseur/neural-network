<details>
<summary><b>Ryzen 9 9950X / Fedora WSL — 2026-11-05 10:43:56 UTC</b></summary>

### Setup

- CPU: AMD Ryzen 9 9950X 16-Core Processor
- Cores: 32 logical
- RAM: 30 GB
- OS: Fedora Linux 44 (WSL)
- Kernel: Linux 6.6.114.1-microsoft-standard-WSL2
- Compiler: g++ (GCC) 16.2.1 20260819 (Red Hat 16.2.1-2), `-std=c++17 -O2 -DNDEBUG -Wall -Wextra -Wpedantic`
- Before: `fef6a15` added a new connect 4 example use case of the network framework
- After: your current files (`7d8566b` + uncommitted changes)
- Timing numbers are medians of 3 rounds, each the median of 5 repeats, with one untimed warmup per case. Run order alternates.
- Both matrix shapes pass the plain dot-product reference (absolute tolerance 1e-9). First-epoch loss and all 87 trained MLP parameters match across versions (absolute/relative tolerance 1e-9).
- No CPU affinity or power controls applied; results describe this machine and workload only.
- Evidence: [before samples](benchmark-runs/20261105T104337Z-gApC7J/before.txt), [after samples](benchmark-runs/20261105T104337Z-gApC7J/after.txt), [metadata](benchmark-runs/20261105T104337Z-gApC7J/metadata.txt), [suite source](benchmark-runs/20261105T104337Z-gApC7J/suite.cpp), [uncommitted library patch](benchmark-runs/20261105T104337Z-gApC7J/source.patch).

### Results

  **2.87× faster**: Matmul 64×784 · 784×128, one multiply

  | Metric | Before | After | Change |
  | :--- | ---: | ---: | :--- |
  | Matmul 100×100, 1000 multiplies (ms) | 240.1 | 293.6 | 1.22× slower |
  | Matmul 100×100 throughput (GFLOP/s) | 8.33 | 6.81 | 1.22× lower |
  | Matmul 64×784 · 784×128, one multiply (ms) | 4.08 | 1.42 | 2.87× faster |
  | Matmul 64×784 · 784×128 throughput (GFLOP/s) | 3.15 | 9.02 | 2.87× higher |
  | Matmul 100×100, allocations per multiply | 4 | 4 | same |
  | Matmul 100×100, memory allocated per multiply (KiB) | 78.16 | 78.16 | same |
  | Small MLP (tensor_mlp network), one epoch (ms) | 10.59 | 5.11 | 2.07× faster |
  | Small MLP, allocations per training step | 547 | 210 | -62% |


</details>
