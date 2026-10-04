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
(`Dense`, `Sigmoid`, `Sequential`), parameters tied to gradients, an SGD optimizer, and automatic
differentiation, so a model can be trained without writing its backward pass by hand.

## Where things stand

M3 (model composition and parity) and the scoped M4 autograd milestone are
complete. The current phase is **M5: the general training stack**. Its first
example-level increment, held-out evaluation for the Tensor MLP, is implemented.

The Tensor example now uses the same seeded 80/20 split as the vector
reference: it updates weights using training rows only, reports training loss
each epoch, and evaluates test loss and accuracy once after training. The
fixed-fixture parity tests continue to compare the framework's forward and
backward calculations with the reference model.

`make test` builds three test programs that AI has helped me write to make sure I cover all test cases:

```text
make test-reference   36 checks   the old MLP on its own, no tensor code linked
make test-tensor     150 checks   Tensor, operations, layers, and autograd
make test-parity     104 checks   the framework compared against the old MLP
```

The parity tests compute gradients three ways (the old model, a manual backward
pass built from plain tensor operations, and autograd) and require all three to
agree to within a 1e-12 tolerence. They also cover a three-layer network and a layer that is
used twice in the same graph.

Everything builds without warnings under `-Wall -Wextra -Wpedantic`, and all
three suites pass under AddressSanitizer and UndefinedBehaviorSanitizer.

## Building

You need a C++17 compiler and GNU Make. I build with GCC 16 on Fedora.

```bash
make                 # builds build/reference_mlp
make run             # runs the old reference model
make test            # all three test suites
make tensor-mlp      # trains the tensor-based model
make graph           # plots the reference model's training loss
```

The Makefile builds with `-O0` for debugging, which makes training slow. For a
optimized run, override the flags:

```bash
make -B tensor-mlp CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -Wpedantic -Iinclude"
```

To run the tests under the sanitizers (on Fedora this needs the `libasan` and
`libubsan` packages downloadable via dnf):

```bash
make -B test CXXFLAGS="-std=c++17 -g3 -O0 -fsanitize=address,undefined -fno-omit-frame-pointer -Wall -Wextra -Wpedantic -Iinclude"
```

## Training a model

This is roughly what `examples/tensor_mlp.cpp` does:

```cpp
std::vector<std::unique_ptr<nnet::Unary_Module>> layers;
layers.push_back(std::make_unique<nnet::Dense>(8, 6));
layers.push_back(std::make_unique<nnet::Sigmoid>());
layers.push_back(std::make_unique<nnet::Dense>(6, 1));
layers.push_back(std::make_unique<nnet::Sigmoid>());

nnet::Sequential model(std::move(layers));
std::vector<nnet::Parameter*> parameters = model.parameters();

for (/* each training row */) {
    // clear models gradients
    model.zeroGrad();
    nnet::Value prediction = model.forward(nnet::makeLeaf(input));
    nnet::backward(nnet::binaryCrossEntropy(prediction, nnet::makeLeaf(target)));
    nnet::sgdOptimizer(parameters, learningRate);
}
```

To evaluate without building a graph, wrap the forward pass in `nnet::NoGrad`:

```cpp
{
    nnet::NoGrad noGrad;
    nnet::Value prediction = model.forward(nnet::makeLeaf(testData));
}
```

## How it's put together

**Tensor** (`include/nnet/core/tensor.h`) owns a contiguous, row-major block of
`double` with a shape and strides. Indexing with `at()` is bounds-checked, and
copies are always deep. There are no views and no broadcasting.

**Operations** (`include/nnet/core/tensor_ops.h`) are free functions on Tensors:
matrix multiply, bias add, sigmoid, binary cross-entropy, transpose, and a few
more. Each differentiable one has a matching gradient function that's checked
against finite differences. Leading dimensions are treated as batch dimensions,
so the same operations work on a single row or a stack of matrices.

**Autograd** (`include/nnet/core/autograd/`) wraps a Tensor in a `Value`. Every
operation on Values leaves behind a small record of what it did and what it was
given. `backward(loss)` walks those records from the loss back to the inputs and
fills in each gradient. A graph can only be used for one backward pass, and
calling it a second time throws rather than counting the gradients twice.

**Layers** (`include/nnet/nn/`) are built on top of autograd. A `Parameter` holds
a value and its gradient together. `Dense` owns a weight and a bias, `Sigmoid`
has no parameters, and `Sequential` chains layers. Layers don't have backward
methods at all; autograd handles that from the operations they use. When
`backward` finishes, each layer's gradients end up in its Parameters, where the
optimizer reads them.

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

Weight initialization uses `rand() / RAND_MAX`. On Linux and macOS, `RAND_MAX`
is 2147483647, but on MSVC it's only 32767, so on Windows every starting weight
would come from about 32 thousand possible values instead of two billion, and
the actual numbers would differ between platforms anyway. There's also no
`srand()` call, so runs can't be seeded. Moving this to a seeded `std::mt19937`
fixes both which will be implemented in the next commit.
The train/test split in `DataFrame` already works that way.

The loss graph (`make graph`) needs matplotlib and a desktop session. It works
on macOS out of the box and on Linux with a display (WSLg under WSL).

## What's next

- Seeded weight initialization, so training runs can be reproduced on purpose.
- Mini-batches instead of one row at a time. Autograd currently builds a whole
  graph per row, which is where most of the training time goes.
- More activations and losses beyond sigmoid and binary cross-entropy.
- Saving and loading trained models.

Longer term, I want this to be a core that can support quite different kinds of
models (language, vision, games, quantitative work) sharing the same operations
and layers, instead of being forced into one inheritance tree. Getting the old
binary classifier to train through it was a checkpoint on the way there, not the
goal.
