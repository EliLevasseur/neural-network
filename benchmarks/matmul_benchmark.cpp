#include <vector>
#include <iostream>
#include <atomic>
#include <new>
#include <chrono>
#include <nnet/core/tensor.h>

using namespace nnet;

// Global trackers for memory allocations
std::atomic<size_t> total_allocated_bytes{0};
 
// Every heap allocation goes through here, which is what makes the counter count.
void* operator new(std::size_t size) {
total_allocated_bytes.fetch_add(size, std::memory_order_relaxed);
if (size == 0) size = 1;
if (void* ptr = std::malloc(size)) return ptr;
throw std::bad_alloc();
}
void operator delete(void* ptr) noexcept { std::free(ptr); }
void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }
 


// HELPER: Generate dummy Tensor data
Tensor create_dummy_tensor(std::size_t batches, std::size_t r, std::size_t c) {
    std::vector<double> d(batches * r * c, 1.5);
    return Tensor({batches, r, c}, std::move(d));
}

void runMultiplication(int runs, std::size_t n) {
for (int i = 0; i < runs; i++) {
        create_dummy_tensor(1, n, n) * create_dummy_tensor(1, n, n);
    }
}

void runNaiveMultiplication(int runs, std::size_t n) {
    for (int i = 0; i < runs; i++) {
        create_dummy_tensor(1, n, n).naiveMatmul(create_dummy_tensor(1, n, n));
    }
}

// BENCHMARKS

void runtime() {
    const std::size_t n = 100;   // the matrices are n x n
    const int runs = 1000;
    const double totalFlops = runs * 2.0 * n * n * n;   // multiply + add for each of n^3 steps

    // OPTIMIZED
    size_t mem_before = total_allocated_bytes.load(std::memory_order_relaxed);
    auto start = std::chrono::high_resolution_clock::now();

    runMultiplication(runs, n);

    auto end = std::chrono::high_resolution_clock::now();
    size_t mem_after = total_allocated_bytes.load(std::memory_order_relaxed);

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = totalFlops / (ms / 1000.0) / 1e9;
    size_t bytes_allocated = mem_after - mem_before;

    std::cout << "OPTIMIZED MATMUL: " << "\nGFLOP/s: " << gflops << "\nMemory Allocated: " << bytes_allocated << "\n" << ms << "ms" << std::endl;

    // NAIVE
    size_t mem_before2 = total_allocated_bytes.load(std::memory_order_relaxed);
    auto start2 = std::chrono::high_resolution_clock::now();

    runNaiveMultiplication(runs, n);

    auto end2 = std::chrono::high_resolution_clock::now();
    size_t mem_after2 = total_allocated_bytes.load(std::memory_order_relaxed);

    double ms2 = std::chrono::duration<double, std::milli>(end2 - start2).count();
    double gflops2 = totalFlops / (ms2 / 1000.0) / 1e9;
    size_t bytes_allocated2 = mem_after2 - mem_before2;

    std::cout << "NAIVE MATMUL: " << "\nGFLOP/s: " << gflops2 << "\nMemory Allocated: " << bytes_allocated2 << "\n" << ms2 << "ms" << std::endl;
}

int main() {
    runtime();
    return 0;
}