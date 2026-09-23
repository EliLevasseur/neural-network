# nnet: a neural network framework built from scratch in C++

This repository is a from-scratch C++ neural-network project with two goals: to
make every calculation inside a neural network understandable and testable, and
to grow a reusable framework whose contracts survive past one example model.

It contains **two systems that coexist on purpose**.

The **reference multilayer perceptron** is a working binary classifier written
with nested `std::vector`. It is finished, verified, and deliberately frozen. Its
job is to be a correctness oracle: an independently trusted source of numbers to
check the new framework against.

The **tensor framework** is the reusable core being built beside it. It is not a
rewrite of the reference model. It is a separate set of components: a `Tensor`,
checked operations, `Parameter`, and `Dense`. They are proven correct by
reproducing the reference model's numbers exactly.

## Current state

Everything below is checked by `make test`, which builds three independent
binaries and runs 157 assertions.

| Suite | Checks | Covers |
|---|---|---|
| `make test-reference` | 36 | the legacy MLP: forward values, loss, analytic and numerical gradients, SGD |
| `make test-tensor` | 83 | `Tensor`, its operations, `DataFrame` conversion, and `Dense` |
| `make test-parity` | 38 | the framework reproducing the reference model's gradients and weight updates |

All three pass warning-clean under `-Wall -Wextra -Wpedantic`, and identically
under a checked-library build (`-D_GLIBCXX_DEBUG -D_GLIBCXX_ASSERTIONS`) and a
release build (`-O2 -DNDEBUG`).

### What parity actually proves

The parity suite builds the same model twice, once with the legacy `Network` and
once from `Dense` layers, pins both to identical weights, and compares:

- every weight gradient and every bias gradient, to a tolerance of `1e-12`;
- every weight and bias after one stochastic gradient descent step;
- the same again for a three-layer network, through a loop that does not know
  how many layers there are.

The reference model's 1000-epoch loss trace also still hashes to the same value
it did before any of this work started, so the oracle itself is untouched.

**What it does not yet prove:** a full training run. Parity covers one sample,
one forward pass, one backward pass, and one update. Multi-epoch training
through the framework does not exist yet.

## Building and running

Requires a C++17 compiler and GNU Make. Built with GCC 16.

```bash
make                 # build build/reference_mlp
make run             # build and run the reference model
make test            # run all three suites
make test-reference  # legacy MLP only, links zero tensor code
make test-tensor     # Tensor, operations, and Dense
make test-parity     # framework against the oracle
make tensor-mlp      # build and run the tensor-based example
make graph           # plot the reference model's training loss (needs matplotlib)
```

The reference model also emits machine-readable output:

```bash
./build/reference_mlp --loss-csv
```

`make test-reference` deliberately links no tensor source file at all, so work in
progress on the framework cannot break the oracle.

## The tensor framework

### Tensor

An owning, contiguous, row-major tensor of `double`.

```cpp
nnet::Tensor matrix({2, 3}, {1, 2, 3, 4, 5, 6});

matrix.rank();        // 2
matrix.numel();       // 6
matrix.shape();       // {2, 3}
matrix.strides();     // {3, 1}
matrix.at({1, 2});    // 6
```

Contracts it holds:

- `at()` requires exactly as many indices as the tensor has axes, and checks
  every one against its own bound.
- Construction rejects a shape whose element count does not match the data, and
  throws `std::overflow_error` rather than wrapping around on an unrepresentable
  shape.
- Copy and move are explicit. Copies are independent.
- `getData()` is const-only, so the storage cannot be resized from outside.
- No views, no aliasing, no broadcasting. Rank zero and zero-sized dimensions
  are rejected rather than supported.

### Operations

Free functions in `nnet/core/tensor_ops.h`, deliberately not methods, so that
what owns state and what computes values stay separate.

**Rank contract:** leading dimensions are batch dimensions and are left
untouched. An operation acts on the last axis, or on the last two for anything
matrix-shaped. Nothing broadcasts.

| Forward | Gradient |
|---|---|
| `sigmoid` | `sigmoidDerivitive` |
| `addBias` | `addBiasGrad` |
| `operator*` (matrix multiply) | `matmulGradWeight`, `matmulGradInput` |
| `binaryCrossEntropy` | `binaryCrossEntropyGrad` |

Also `transpose`, `sum`, `sumLeadingDimensions`, `fill`, `zeros`, and the
`Tensor` operators `+`, `-`, scalar `*`, and `inplaceMultiplication`.

Every gradient above is checked against a central finite difference, not just
against a hand-computed fixture.

Matrix multiply takes either two operands of equal rank, pairing each matrix
with its own, or a higher-rank left operand against a single rank-2 right
operand that every group shares. The second form is what a layer applied across
grouped data needs.

### Parameter

One trainable thing: a value and the gradient belonging to it, in one object so
they cannot be mismatched.

```cpp
nnet::Parameter weight(nnet::Tensor({2, 2}, {0.1, 0.4, -0.2, 0.3}));
weight.value;   // the numbers
weight.grad;    // same shape, zero-filled at birth
```

Copying a `Parameter` is a compile error. An optimizer has to update the exact
object the forward pass read, so a copy would be silently useless.

### Dense

One fully connected layer. Owns its weight and bias, applies no activation, and
caches nothing.

```cpp
nnet::Dense layer(3, 2);                  // 3 inputs, 2 outputs

nnet::Tensor output = layer.forward(input);
nnet::Tensor inputGradient = layer.backward(savedInput, gradientFromAbove);

for (nnet::Parameter* p : layer.parameters()) { /* weight, then bias */ }
```

`forward` is `const`, which is load-bearing rather than decorative: a const
method cannot assign to a member, so the layer physically cannot stash values
between calls. `backward` takes the forward pass's input back as an argument for
the same reason.

The activation lives outside the layer, chosen by the caller:

```cpp
nnet::Tensor activation = input;
for (const nnet::Dense& layer : layers) {
    activation = nnet::sigmoid(layer.forward(activation));
}
```

## The reference model

Still buildable, still the oracle. Weights are `weights[node][input]`, one
`Layer` per entry, sigmoid throughout, binary cross-entropy, per-sample SGD.

```cpp
DataFrame dataFrame("data/binary_test.csv", 3);
const auto split = dataFrame.trainTestSplit(0.8, 42);

Network network({3, 5, 3, 1});
Trainer trainer(network, 0.09);
trainer.fit(1000, split.XTrain, split.yTrain);

const auto predictions = network.predict(split.XTest);
const double accuracy = trainer.getAccuracy(predictions, split.yTest);
```

Its verified behaviour covers only the repository's fixed test fixtures. It is
not a safe general-purpose API: malformed files, ragged rows, and mismatched
shapes are not all checked, and some checks are assertions that vanish in a
release build.

## Project structure

```text
neural-network/
|-- data/                       small deterministic CSV fixtures
|-- include/
|   |-- dataframe.h             CSV loading and train/test split
|   |-- network.h               legacy MLP
|   |-- training.h              legacy trainer
|   `-- nnet/
|       |-- core/
|       |   |-- tensor.h        the Tensor value type
|       |   `-- tensor_ops.h    operations and their gradients
|       `-- nn/
|           |-- parameter.h     a value plus its gradient
|           `-- dense.h         one fully connected layer
|-- src/                        implementations, mirroring include/
|-- examples/
|   |-- reference_mlp.cpp       the legacy model, build/reference_mlp
|   `-- tensor_mlp.cpp          Dense layers end to end, build/tensor_mlp
|-- tests/                      three independent suites and their entry points
|-- visualizations/graphing.py  training-loss plot
`-- Makefile
```

## Platform support

Developed and verified on Linux and macOS. Windows is not supported yet, and a
Visual Studio build is planned.

**The build assumes a POSIX shell.** Every target calls `mkdir -p` and runs its
binary as `./build/name`, neither of which works in `cmd.exe`. That alone stops
`make` from running natively on Windows, which is why a Visual Studio project is
the route in rather than patching the Makefile.

**Weight initialization is not portable.** Both weight generators call `rand()`
and divide by `RAND_MAX`:

```cpp
weight = ((double)rand() / RAND_MAX) - 0.5;
```

`RAND_MAX` is implementation-defined. The standard only promises it is at least
32767, and the real values differ sharply:

| Toolchain | `RAND_MAX` | Distance between adjacent weights |
|---|---|---|
| glibc and macOS libc | 2147483647 | 0.00000000047 |
| MSVC | 32767 | 0.0000305 |

So on Windows every weight is drawn from a pool of 32768 values instead of two
billion, roughly 65000 times coarser. Drawing 4096 weights from that pool
produces around 240 exact duplicates, where on Linux duplicates are effectively
impossible.

The generators themselves also differ, so the same code produces entirely
different starting weights on each platform. There is no `srand()` call anywhere
either, meaning the sequence is fixed at the standard's default seed of 1 and
cannot be chosen. Replacing this with a seeded `std::mt19937` and an explicit
distribution fixes both problems at once, and is listed under the next steps
below. `DataFrame::trainTestSplit` already does exactly that, so the data split
is portable even though the weights are not.

**The training-loss graph needs a desktop session.** `make graph` pipes the
model's output into matplotlib, which needs the package installed and an
interactive display:

- macOS works with no setup beyond installing matplotlib.
- Linux needs a display. Under WSL that means WSLg.
- Windows has no target for it, since the pipeline is a shell pipe calling
  `python3`, which on Windows is usually `python` or `py`.

Nothing in `visualizations/graphing.py` is platform-specific. What varies is the
environment around it.

## Not built yet

- A training loop through the framework. Forward, backward, and the update rule
  all exist and are verified, but nothing runs them over epochs yet.
- An optimizer as its own component. The update rule is three verified lines
  inside the parity test.
- Automatic differentiation. Every gradient is currently applied by hand.
- A shared base class for layers. There is only one kind of layer, so there is
  nothing to unify yet.
- Seeded weight initialization. Weights come from the global random function,
  which makes training runs impossible to reproduce deliberately.
- Activations other than sigmoid, losses other than binary cross-entropy,
  mini-batching, serialization, and any optimized or accelerated backend.

## Where this is going

**Immediately next, in order:**

1. A training loop and an optimizer component. Forward, backward, and the update
   rule all exist and are checked against the oracle. Nothing runs them over
   epochs yet, so no model has actually been trained through the framework.
2. Seeded weight initialization, so two runs can be compared deliberately rather
   than by accident of the C library's default random sequence.
3. Reverse-mode automatic differentiation. Applying four gradient rules by hand
   is fine. Applying forty is where hand-rolled frameworks fall over, and it is
   the thing that makes every later model affordable.

**Long term goals:**

- parameters, modules, losses, optimizers and training utilities as separate
  components rather than one fused trainer;
- deterministic data handling and metrics;
- serialization, so a trained model can be saved and loaded;
- embedding, convolution, normalization, attention, and recurrent building
  blocks, each added when its prerequisites actually exist;
- a readable CPU reference kernel kept alongside optimized ones, with optional
  accelerator backends behind the same tested operation contracts;
- reproducible testing, benchmarking, and documentation throughout.

The target is a core that can carry at least two structurally different model
families. Language, vision, game, and quantitative models should be able to
share operations and components without being forced into one inheritance tree.
Reproducing the binary classifier is a verification step along the way, not the
ceiling.
