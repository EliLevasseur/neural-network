// The benchmarks behind every entry in info/PERFORMANCE.md.
//
// It only uses parts of the library that have existed since the first commit
// being compared (Tensor, operator*, Dense, Sequential, trainEpoch, SGD), so
// scripts/benchmark.sh can build this same file against an old commit and a
// new one and compare them. If you add a benchmark that needs a newer
// function, older commits won't compile it.
//
// Output is one "name value" line per measurement, for the script to read:
//   ./bench_suite            5 repeats per measurement, median reported
//   ./bench_suite 3          3 repeats (faster, noisier)

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <new>
#include <random>
#include <vector>

#include "nnet/core/tensor.h"
#include "nnet/core/tensor_ops.h"
#include "nnet/nn/dense.h"
#include "nnet/nn/module.h"
#include "nnet/nn/sequential.h"
#include "nnet/optim/optimizer.h"
#include "nnet/train/trainer.h"

using nnet::Tensor;

// ---------------------------------------------------------------------------
// Allocation counting. Every heap allocation in the program goes through
// operator new, so replacing it counts them all. Read the counters before and
// after the code being measured and subtract.
// ---------------------------------------------------------------------------

static std::size_t allocationCount = 0;
static std::size_t allocatedBytes = 0;

// noinline: if GCC inlines these, it sees malloc/free paired with new/delete
// and warns about a mismatch that isn't really there.
#if defined(__GNUC__)
#define NNET_BENCH_NOINLINE __attribute__((noinline))
#else
#define NNET_BENCH_NOINLINE
#endif

NNET_BENCH_NOINLINE void* operator new(std::size_t size) {
    allocationCount++;
    allocatedBytes += size;
    if (size == 0) {
        size = 1;
    }
    if (void* memory = std::malloc(size)) {
        return memory;
    }
    throw std::bad_alloc();
}
NNET_BENCH_NOINLINE void operator delete(void* memory) noexcept { std::free(memory); }
NNET_BENCH_NOINLINE void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }

// ---------------------------------------------------------------------------
// Timing
// ---------------------------------------------------------------------------

using Clock = std::chrono::steady_clock;

static volatile double sink = 0.0;  // the compiler can't skip work that writes here

template <typename Function>
double timeMs(Function&& function) {
    const auto start = Clock::now();
    function();
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

// One untimed warm-up, then the median of `repeats` timed runs.
template <typename Function>
double medianMs(int repeats, Function&& function) {
    function();
    std::vector<double> times;
    for (int i = 0; i < repeats; i++) {
        times.push_back(timeMs(function));
    }
    std::sort(times.begin(), times.end());
    return times[times.size() / 2];
}

static void report(const char* name, double value) {
    std::printf("%s %.6g\n", name, value);
}

// ---------------------------------------------------------------------------
// Data
// ---------------------------------------------------------------------------

static Tensor randomTensor(const Tensor::Shape& shape, std::mt19937& generator, double low, double high) {
    std::size_t total = 1;
    for (const std::size_t dimension : shape) {
        total *= dimension;
    }
    std::uniform_real_distribution<double> distribution(low, high);
    std::vector<double> values(total);
    for (double& value : values) {
        value = distribution(generator);
    }
    return Tensor(shape, std::move(values));
}

// Targets for `rows` examples: one 0/1 column when classes == 1, otherwise
// one-hot rows.
static Tensor randomTargets(std::size_t rows, std::size_t classes, std::mt19937& generator) {
    const std::size_t width = classes == 1 ? 1 : classes;
    std::vector<double> values(rows * width, 0.0);
    std::uniform_int_distribution<std::size_t> label(0, classes == 1 ? 1 : classes - 1);
    for (std::size_t row = 0; row < rows; row++) {
        const std::size_t picked = label(generator);
        if (classes == 1) {
            values[row] = static_cast<double>(picked);
        } else {
            values[row * width + picked] = 1.0;
        }
    }
    return Tensor({rows, width}, std::move(values));
}

// Plain dot-product matmul, to check operator* still gives the right answer.
static double maxErrorAgainstReference(const Tensor& left, const Tensor& right, const Tensor& product) {
    const std::size_t rows = left.shape()[0];
    const std::size_t inner = left.shape()[1];
    const std::size_t cols = right.shape()[1];
    const std::vector<double>& a = left.getData();
    const std::vector<double>& b = right.getData();
    const std::vector<double>& c = product.getData();
    double worst = 0.0;
    for (std::size_t i = 0; i < rows; i++) {
        for (std::size_t j = 0; j < cols; j++) {
            double total = 0.0;
            for (std::size_t k = 0; k < inner; k++) {
                total += a[i * inner + k] * b[k * cols + j];
            }
            worst = std::max(worst, std::abs(total - c[i * cols + j]));
        }
    }
    return worst;
}

// ---------------------------------------------------------------------------
// Benchmarks
// ---------------------------------------------------------------------------

// 1000 multiplies of 100x100 matrices, the same work as the first entry in
// PERFORMANCE.md. The inputs are built once, before timing.
static void matmulSquare(int repeats) {
    std::mt19937 generator(1);
    const Tensor a = randomTensor({100, 100}, generator, -1.0, 1.0);
    const Tensor b = randomTensor({100, 100}, generator, -1.0, 1.0);
    const int multiplies = 1000;

    report("matmul100_max_error", maxErrorAgainstReference(a, b, a * b));

    const double ms = medianMs(repeats, [&] {
        for (int i = 0; i < multiplies; i++) {
            sink = sink + (a * b).getData()[0];
        }
    });
    report("matmul100_ms", ms);
    report("matmul100_gflops", multiplies * 2.0 * 100 * 100 * 100 / (ms * 1e6));

    const std::size_t countBefore = allocationCount;
    const std::size_t bytesBefore = allocatedBytes;
    {
        const Tensor product = a * b;
        sink = sink + product.getData()[0];
    }
    report("matmul100_allocs", static_cast<double>(allocationCount - countBefore));
    report("matmul100_kb", static_cast<double>(allocatedBytes - bytesBefore) / 1024.0);
}

// A batch of 64 rows times a 784x128 weight: the size of a real network
// layer, where loop order and cache use matter.
static void matmulLayer(int repeats) {
    std::mt19937 generator(2);
    const Tensor x = randomTensor({64, 784}, generator, 0.0, 1.0);
    const Tensor w = randomTensor({784, 128}, generator, -0.1, 0.1);
    const int multiplies = 20;

    const double ms = medianMs(repeats, [&] {
        for (int i = 0; i < multiplies; i++) {
            sink = sink + (x * w).getData()[0];
        }
    }) / multiplies;
    report("matmul_layer_ms", ms);
    report("matmul_layer_gflops", 2.0 * 64 * 784 * 128 / (ms * 1e6));
}

using LayerList = std::vector<std::unique_ptr<nnet::Unary_Module>>;

// Times training epochs of `model` and reports the time per step, the
// allocations per step, and the loss (which should not change when an
// optimization only changes speed).
static void trainingBenchmark(const char* name, nnet::Sequential& model, const Tensor& inputs,
                              const Tensor& targets, std::size_t batchSize, double learningRate,
                              nnet::LossFunction loss, int repeats) {
    const std::size_t rows = inputs.shape()[0];
    const double steps = static_cast<double>((rows + batchSize - 1) / batchSize);
    nnet::SGDOptimizer optimizer(model.parameters(), learningRate);
    std::mt19937 shuffle(42);

    // The first epoch's loss, from a fixed starting point, is the
    // correctness check. Every later epoch starts from different weights.
    const double firstLoss = nnet::trainEpoch(model, inputs, targets, batchSize, optimizer, loss, shuffle);

    const double epochMs = medianMs(repeats, [&] {
        sink = sink + nnet::trainEpoch(model, inputs, targets, batchSize, optimizer, loss, shuffle);
    });

    const std::size_t countBefore = allocationCount;
    sink = sink + nnet::trainEpoch(model, inputs, targets, batchSize, optimizer, loss, shuffle);
    const double allocations = static_cast<double>(allocationCount - countBefore);

    char key[64];
    std::snprintf(key, sizeof(key), "%s_epoch_ms", name);
    report(key, epochMs);
    std::snprintf(key, sizeof(key), "%s_step_ms", name);
    report(key, epochMs / steps);
    std::snprintf(key, sizeof(key), "%s_allocs_per_step", name);
    report(key, allocations / steps);
    std::snprintf(key, sizeof(key), "%s_loss", name);
    std::printf("%s %.12g\n", key, firstLoss);
}

// The tensor_mlp example's network: 8 -> 6 -> 4 -> 1 with sigmoids, one row
// per step, on random data the same shape as tensor_mlp's training split.
// Tiny matrices, so this measures per-step overhead (autograd, allocations).
static void smallNetwork(int repeats) {
    std::mt19937 data(3);
    const Tensor inputs = randomTensor({1639, 8}, data, -1.0, 1.0);
    const Tensor targets = randomTargets(1639, 1, data);

    std::mt19937 weights(42);
    const std::vector<std::size_t> shape = {8, 6, 4, 1};
    LayerList layers;
    for (std::size_t i = 0; i + 1 < shape.size(); i++) {
        layers.push_back(std::make_unique<nnet::Dense>(shape[i], shape[i + 1], weights));
        layers.push_back(std::make_unique<nnet::Sigmoid>());
    }
    nnet::Sequential model(std::move(layers));
    trainingBenchmark("mlp_small", model, inputs, targets, 1, 0.5, nnet::binaryCrossEntropy, repeats);
}

int main(int argc, char** argv) {
    const int repeats = argc > 1 ? std::max(1, std::atoi(argv[1])) : 5;
    matmulSquare(repeats);
    matmulLayer(repeats);
    smallNetwork(repeats);
    return 0;
}