# Performance

Results are grouped by **test**, with a dated run for each machine underneath.
Keep the same `-t` title when repeating a comparison on another machine. Older
runs stay in place; each successful run adds another entry.

From the repository root:

```bash
CXX=g++ ./scripts/benchmarks.sh main -t "Optimization branch" -m "My PC / Fedora WSL"
```

This compares `main` with your current library files, including uncommitted edits.
Supply a second commit to compare two saved versions. Defaults: three rounds,
five timed repeats and one warmup per benchmark. Use `-r` and `-n` to change them.
Requires Bash, Git, a C++17 compiler, Python 3 and standard Unix tools.

The script appends here automatically. Raw samples, exact suite source, commit
IDs, compiler settings and the uncommitted library patch stay in
`info/benchmark-runs/`. Failed runs retain their evidence but add no performance
claim here. Commit successful evidence alongside this log when sharing results.

The current suite measures 100×100 matrix multiplication, a 64×784 · 784×128
dense-layer multiplication, and training an 8→6→4→1 sigmoid MLP on seeded
synthetic data. Timing excludes input generation; training epochs advance the
same seeded model on each side. Sample keys ending in `layer_20_ms` time 20
multiplies; the table divides that duration by 20. Allocation figures count
ordinary `operator new` calls/bytes, not peak resident memory.

Speed ratios apply to the listed workload and machine. Overlapping observed
ranges are flagged as possibly noisy, not treated as a statistical test.

## Matrix Multiplication Optimization

<details>
<summary><b>Mac — 2026-10-29 (historical benchmark)</b></summary>

### Setup

- CPU: Intel Core i7-1068NG7 @ 2.30 GHz
- Compiler: Apple Clang 21, `-std=c++17 -O2`
- Program: `benchmarks/matmul_benchmark.cpp`
- Work: 1,000 multiplies of 100×100 matrices (2 billion arithmetic operations)
- Before: `naiveMatmul` (i-j-k); after: `operator*` (i-k-j)

### Results

| Metric | Before | After |
| :--- | ---: | ---: |
| Total time (ms) | 1035.68 | 333.286 |
| Total allocated bytes | 240240000 | 240240000 |

About **3.1× faster** in this recorded run. The original entry called the byte
count “Memory Allocations”; the program actually measures allocated bytes.

These original measurements are preserved. Repeats, warmups, raw samples and
exact commit IDs were not recorded. This older program creates inputs inside
the timed loop, so its absolute timings are not directly comparable to the
current suite above.

</details>

## Optimization branch

<details>
<summary><b>Ryzen 9 9950X / Fedora WSL repeatability check — 2026-11-05 10:49:10 UTC</b></summary>

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
- Evidence: [before samples](benchmark-runs/20261105T104851Z-kCqt5U/before.txt), [after samples](benchmark-runs/20261105T104851Z-kCqt5U/after.txt), [metadata](benchmark-runs/20261105T104851Z-kCqt5U/metadata.txt), [suite source](benchmark-runs/20261105T104851Z-kCqt5U/suite.cpp), [uncommitted library patch](benchmark-runs/20261105T104851Z-kCqt5U/source.patch).

### Results

  **2.98× faster**: Matmul 64×784 · 784×128, one multiply

  | Metric | Before | After | Change |
  | :--- | ---: | ---: | :--- |
  | Matmul 100×100, 1000 multiplies (ms) | 258.5 | 297.1 | 1.15× slower (within noise) |
  | Matmul 100×100 throughput (GFLOP/s) | 7.74 | 6.73 | 1.15× lower (within noise) |
  | Matmul 64×784 · 784×128, one multiply (ms) | 4.24 | 1.42 | 2.98× faster |
  | Matmul 64×784 · 784×128 throughput (GFLOP/s) | 3.03 | 9.04 | 2.98× higher |
  | Matmul 100×100, allocations per multiply | 4 | 4 | same |
  | Matmul 100×100, memory allocated per multiply (KiB) | 78.16 | 78.16 | same |
  | Small MLP (tensor_mlp network), one epoch (ms) | 10.80 | 5.02 | 2.15× faster |
  | Small MLP, allocations per training step | 547 | 210 | -62% |

  "Within noise" means the before and after runs overlapped, so that difference may not be real.

</details>

