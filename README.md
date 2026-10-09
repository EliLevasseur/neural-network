# Deep Learning Framework

A neural network framework I'm writing from scratch in C++, mainly to understand
how every piece actually works: tensors, gradients, layers, training. No outside
math or ML libraries.

The repo has two separate implementations on purpose.

The first is a small multilayer perceptron written with nested `std::vector`. It
was the first thing I got working, and it's now frozen. I use it as a reference:
when the newer code produces a number, I check it against what this model
produces for the same weights and input.

The second is the actual framework, built around a `Tensor` type. It has components
(`Dense`, `ReLU`, `Sigmoid`, `Sequential`), parameters tied to gradients, SGD, momentum and Adam
optimizers, and automatic differentiation, so a model can be trained without writing its backward
pass by hand.

## Recognizing handwritten digits

The framework can train a network on MNIST, the standard set of 28x28
handwritten digits. I train on 50,000 of the training images, keep the other
10,000 aside as a validation set to watch during training, and only score the
10,000 test images once, at the very end. A network with one hidden layer of
128 units (784 -> 128 -> 10, ReLU, softmax cross-entropy, mini-batches of 64)
gets 97.07% of the test images right after 10 passes over the training set.
Each pass takes about 6.5 seconds on one core of a Ryzen 9 9950X.

```bash
make mnist-data   # downloads MNIST into data/mnist/ and checks its checksums
make mnist        # trains the network and tests it
```

Everything is seeded, so apart from the timings it prints the same numbers
every time I run it:

```text
Before training: validation accuracy 10.52% (guessing)
Epoch  1 | training loss 0.3799 | validation accuracy 92.48% | 6.4 s
Epoch  2 | training loss 0.2064 | validation accuracy 94.55% | 6.4 s
...
Epoch 10 | training loss 0.0609 | validation accuracy 97.12% | 6.2 s

Final test accuracy: 97.07% (9707 of 10000 images it never trained on)
```

At the end it draws some test images in the terminal with its guess, how sure
it was, and the right answer, plus a few of the ones it gets wrong:

```text
        .--:.
        +#*#%@%%%%%%%%%-
              . ....:@@:
                   :@@:
                  -@%.
                 :@%.
                +@#.
               %@+
             -@@-
            -@@@.
            :==
  guess 7 (99.9%) true 7
```

## Where things stand

Both examples split a validation set off their training data, watch it during
training, and only evaluate the test set once at the end. The fixed-fixture
parity tests still compare the framework's forward and backward calculations
with the reference model.

`make test` builds three test programs that AI has helped me write to make sure I cover all test cases:

```text
make test-reference   36 checks   the old MLP on its own, no tensor code linked
make test-tensor     225 checks   Tensor, operations, layers, autograd, training, MNIST loading, checkpoints
make test-parity     104 checks   the framework compared against the old MLP
```

The parity tests compute gradients three ways (the old model, a manual backward
pass built from plain tensor operations, and autograd) and require all three to
agree to within a 1e-12 tolerance. They also cover a three-layer network and a layer that is
used twice in the same graph.

The optimizers have their own test program, `tests/optimizer_test.cpp` (46
checks), which CTest runs automatically, or can be built with the command at the top of the file.

Everything builds without warnings under `-Wall -Wextra -Wpedantic`, and all
three suites pass under AddressSanitizer and UndefinedBehaviorSanitizer.

## Building

Use CMake 3.20+, GCC with C++17 support, and GNU Make inside Fedora WSL:

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

CMake builds reusable libraries and separate examples/tests. CTest runs all four
suites, including the optimizer checks: 411 checks in total. The reference
executable remains independent of Tensor/autograd.

Use `checked`, `release`, or `asan` in place of `debug` for checked libstdc++,
optimized training, or AddressSanitizer/UndefinedBehaviorSanitizer. Build
directories are separate and each includes `compile_commands.json`.

```bash
cmake --preset release
cmake --build --preset release
./build/cmake-release/tensor_mlp
make mnist-data
./build/cmake-release/mnist
```

Run examples from the repository root so their relative dataset paths resolve.
Use Release for MNIST; the CMake Debug build deliberately stays unoptimized.
Builds and tests do not download datasets.

See [the CMake guide](cmake/README.md) for targets, presets, installation, and a
separate example project that links the installed framework using
`find_package(nnet CONFIG REQUIRED)`.

The original Make workflow remains available:

```bash
make                 # builds build/reference_mlp
make run             # runs the old reference model
make test            # reference, Tensor and parity suites
make tensor-mlp      # trains the tensor-based model
make mnist-data      # downloads MNIST and checks its checksums
make mnist           # trains the digit classifier (built with -O2)
make graph           # plots the reference model's training loss
```

Make still builds three suites; CTest includes the standalone optimizer suite
as its fourth. With Make, supply `CXXFLAGS` and `-B` to change build modes;
with CMake, use the presets. GCC Debug, checked, Release and ASan/UBSan CMake
builds are verified locally. Hosted CI and Clang verification remain pending.

## Training a model

This is roughly what `examples/mnist.cpp` does:

```cpp
std::mt19937 weightGenerator(42);
std::vector<std::unique_ptr<nnet::Unary_Module>> layers;
layers.push_back(std::make_unique<nnet::Dense>(784, 128, weightGenerator));
layers.push_back(std::make_unique<nnet::ReLU>());
layers.push_back(std::make_unique<nnet::Dense>(128, 10, weightGenerator));
nnet::Sequential model(std::move(layers));

std::mt19937 splitGenerator(42);
nnet::TrainValidationSplit train = nnet::splitOffValidation(mnist.images, mnist.labels, 10000, splitGenerator);

nnet::SGDOptimizer optimizer(model.parameters(), 0.1);
std::mt19937 shuffleGenerator(42);
for (int epoch = 0; epoch < epochs; epoch++) {
    double trainingLoss = nnet::trainEpoch(model, train.trainInputs, train.trainTargets, 64, optimizer,
                                           nnet::softmaxCrossEntropy, shuffleGenerator);
    double validationAccuracy = nnet::accuracy(model, train.validationInputs, train.validationTargets);
}
double testAccuracy = nnet::accuracy(model, test.images, test.labels);
```

Swapping `SGDOptimizer` for `MomentumOptimizer` or `AdamOptimizer` is the only
change needed to train with those instead.

`trainEpoch` shuffles the rows, cuts them into batches, and runs one training
step per batch. Written out by hand, one step is:

```cpp
model.zeroGrad();
nnet::Value scores = model.forward(nnet::makeLeaf(batchImages));
nnet::backward(nnet::softmaxCrossEntropy(scores, nnet::makeLeaf(batchLabels)));
optimizer.step();
```

To evaluate without building a graph, wrap the forward pass in `nnet::NoGrad`:

```cpp
{
    nnet::NoGrad noGrad;
    nnet::Value prediction = model.forward(nnet::makeLeaf(test.images));
}
```

A trained model can be saved and loaded again. The file only holds the numbers,
so the model has to be built the same way in code before loading:

```cpp
nnet::saveParameters(model, "mnist.ckpt");
nnet::loadParameters(model, "mnist.ckpt");
```

## How it's put together

**Tensor** (`include/nnet/core/tensor.h`) owns a contiguous, row-major block of
`double` with a shape and strides. Indexing with `at()` is bounds-checked, and
copies are always deep. There are no views and no broadcasting.

**Operations** (`include/nnet/core/tensor_ops.h`) are free functions on Tensors:
matrix multiply, bias add, sigmoid, ReLU, softmax, binary and softmax
cross-entropy, transpose, and a few more. Each differentiable one has a matching gradient function that's checked
against finite differences. Leading dimensions are treated as batch dimensions,
so the same operations work on a single row or a stack of matrices.

**Autograd** (`include/nnet/core/autograd/`) wraps a Tensor in a `Value`. Every
operation on Values leaves behind a small record of what it did and what it was
given. `backward(loss)` walks those records from the loss back to the inputs and
fills in each gradient. A graph can only be used for one backward pass, and
calling it a second time throws rather than counting the gradients twice.

**Layers** (`include/nnet/nn/`) are built on top of autograd. A `Parameter` holds
a value and its gradient together. `Dense` owns a weight and a bias, with the
weight's starting values drawn from a seeded generator (He initialization).
`Sigmoid` and `ReLU` have no parameters, and `Sequential` chains layers. Layers don't have backward
methods at all; autograd handles that from the operations they use. When
`backward` finishes, each layer's gradients end up in its Parameters, where the
optimizer reads them.

**Optimizers** (`include/nnet/optim/optimizer.h`) all share one small
`Optimizer` base class with a `step()`. SGD, momentum and Adam each keep
whatever they need between steps (momentum a velocity per weight, Adam a
velocity and an average of squared gradients), and skip frozen parameters.

**Training** (`include/nnet/train/trainer.h`) is a few plain functions.
`trainEpoch` does one shuffled pass over the data in mini-batches with any
optimizer, `splitOffValidation` sets aside a validation set, and
`evaluateLoss` and `accuracy` measure a model without building a graph. They
take inputs and targets as Tensors, so they don't care where the data came
from: `loadMnist` (`include/nnet/data/mnist.h`) reads the MNIST files into
Tensors, and `DataFrame` does the same for CSV files. Nothing has to go
through the trainer, though. A model that doesn't fit it can call `forward`,
`backward` and `step()` itself.

**Checkpoints** (`include/nnet/nn/checkpoint.h`) save each parameter's name,
shape and values to a binary file, and refuse to load a file that doesn't
match the model.

## The reference model

The old model is still in `include/network.h`, `include/training.h`, and
`examples/reference_mlp.cpp`. It uses sigmoid layers, binary cross-entropy, and
per-sample SGD, with weights stored as `weights[node][input]`.

```cpp
DataFrame data("data/binary_test.csv", 3);
const auto split = data.trainTestSplit(0.8, 42);

Network network({3, 5, 3, 1});
Trainer trainer(network, 0.09);
trainer.fit(1000, split.XTrain, split.yTrain);
```

It's only been tested against the fixtures in this repo. It doesn't validate
input files or shapes carefully, and some of its checks are `assert`s that
disappear in a release build, so it isn't meant for general use.

## Platform notes

So far this has only been built and tested on Linux and macOS. I plan to add a
Visual Studio build for Windows. There are two things to know before that:

The Makefile assumes a Unix shell (`mkdir -p`, `./build/...`), so `make` won't
run from `cmd.exe`.

The tensor framework draws its starting weights from a seeded `std::mt19937`,
the same way the train/test split in `DataFrame` works, so a run starts from
the same weights every time. The legacy reference model still uses
`rand() / RAND_MAX`, because its output is the baseline everything else gets
checked against. On Linux and macOS `RAND_MAX` is 2147483647, but on MSVC it's
only 32767, so that model's starting weights would be much coarser on Windows.

The loss graph (`make graph`) needs matplotlib and a desktop session. It works
on macOS out of the box and on Linux with a display (WSLg under WSL).

## What's next

- Measuring where the training time actually goes, then making it faster.
- More layer types and losses than the ones I have so far.

Longer term, I want this to be a core that can support quite different kinds of
models (language, vision, games, quantitative work) sharing the same operations
and layers, instead of being forced into one inheritance tree. Getting the old
binary classifier to train through it was a checkpoint on the way there, not the
goal.
