#include <vector>
#include <iostream>
#include <atomic>
#include <string_view>
#include <new>
#include <chrono>
#include <utility>
#include <nnet/core/tensor.h>

using namespace nnet;

// Global trackers for memory allocations
std::atomic<size_t> total_allocated_bytes{0};

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

// WORKLOADS: Now take pre-allocated tensors so setup doesn't pollute benchmarks
void runMultiplication(int runs, const Tensor& a, const Tensor& b) {
    for (int i = 0; i < runs; i++) {
        auto res = a * b;
    }
}

void runNaiveMultiplication(int runs, const Tensor& a, const Tensor& b) {
    for (int i = 0; i < runs; i++) {
        auto res = a.naiveMatmul(b);
    }
}

// GENERIC BENCHMARK HELPER
template <typename Func, typename... Args>
void benchmark(std::string_view label, double totalFlops, Func&& func, Args&&... args) {
    size_t mem_before = total_allocated_bytes.load(std::memory_order_relaxed);
    auto start = std::chrono::high_resolution_clock::now();

    std::forward<Func>(func)(std::forward<Args>(args)...);

    auto end = std::chrono::high_resolution_clock::now();
    size_t mem_after = total_allocated_bytes.load(std::memory_order_relaxed);

    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double gflops = totalFlops / (ms / 1000.0) / 1e9;
    size_t bytes_allocated = mem_after - mem_before;

    std::cout << label << ":\n"
              << "  GFLOP/s: " << gflops << "\n"
              << "  Memory Allocated: " << bytes_allocated << " bytes\n"
              << "  Time: " << ms << " ms\n\n";
}

void runtime() {
    const int runs = 1000;

    // [128, 512] × [512, 128] multipliction run.

    const std::size_t matrix1_rows = 128;
    const std::size_t matrix1_cols = 512;

    const std::size_t matrix2_rows = 512;
    const std::size_t matrix2_cols = 128;

    const std::size_t batches = 1;
    const std::size_t batches2 = 1;

    const double squareFlops = runs * 2.0 * matrix1_rows * matrix1_cols * matrix2_cols;
    Tensor sqA = create_dummy_tensor(batches, matrix1_rows, matrix1_cols);
    Tensor sqB = create_dummy_tensor(batches2, matrix2_rows, matrix2_cols);

    benchmark("OPTIMIZED MATMUL (128x512)*(512x128)", squareFlops, runMultiplication, runs, sqA, sqB);
    benchmark("NAIVE MATMUL (128x512)*(512x128)", squareFlops, runNaiveMultiplication, runs, sqA, sqB);
}

int main() {
    runtime();
    return 0;
}
