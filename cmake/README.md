# Building with CMake

You need CMake 3.20 or newer, GNU Make, and a C++17 compiler. I build and test
with GCC on Fedora under WSL; other compilers and platforms haven't been checked
yet. Nothing gets downloaded when you build nnet itself. Run the commands below
from the repository root.

## Quick start

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release
./build/cmake-release/tensor_mlp
```

That configures the build, compiles everything, runs the tests, and trains the
small example model. The tests are four programs with 420 checks between them:
the old reference model on its own (36), the Tensor framework (234), the
framework compared against the reference model (104), and the optimizers (46).

## Build modes

Each preset builds into its own folder, so you can keep several around at once.
Swap the name in the first three commands above to use a different one.

| Preset | What it's for | Folder |
|---|---|---|
| debug | Stepping through code: `-g3 -O0` | build/cmake-debug |
| release | Training and timing: `-O2 -DNDEBUG` | build/cmake-release |
| checked | Debug, plus libstdc++ checks on every container access | build/cmake-checked |
| asan | Debug, plus AddressSanitizer and UndefinedBehaviorSanitizer | build/cmake-asan |

`checked` and `asan` are slower, but they catch out-of-bounds reads and other
memory bugs that a normal build can quietly get away with. `checked` changes the
standard library's layout, so anything linked against a checked build has to be
built the same way. The sanitizers need the compiler's ASan and UBSan runtime
libraries installed.

Every build folder gets a `compile_commands.json` for editors. If you want your
own presets, put them in `CMakeUserPresets.json`, which git ignores.

## Using nnet in your own project

There are two ways to do it.

### Let CMake download it

Add this to your project's `CMakeLists.txt`:

```cmake
include(FetchContent)
FetchContent_Declare(nnet
    GIT_REPOSITORY https://github.com/EliLevasseur/neural-network.git
    GIT_TAG v0.2.0)
FetchContent_MakeAvailable(nnet)

target_link_libraries(your_program PRIVATE nnet::framework)
```

CMake clones nnet the first time you configure and builds it along with your
program. Pulled in this way, nnet only builds its libraries, not its own tests
or examples. `GIT_TAG` pins the version, so a later change to nnet can't break
your build. To build against a local clone instead, for example while you're
changing nnet itself, configure with
`-DFETCHCONTENT_SOURCE_DIR_NNET=/path/to/neural-network`.
[GLM](https://github.com/EliLevasseur/LLM) is set up this way.

### Install it once

```bash
cmake --preset release
cmake --build --preset release
cmake --install build/cmake-release --prefix $HOME/libs/nnet
```

Then in your project:

```cmake
find_package(nnet CONFIG REQUIRED)
target_link_libraries(your_program PRIVATE nnet::framework)
```

and configure it with `-DCMAKE_PREFIX_PATH=$HOME/libs/nnet` so CMake can find
it. The install folder can be moved afterwards; it doesn't point back at the
source checkout.

### Which library to link

- `nnet::framework` is what most programs want: tensors, the operations,
  autograd, the layers, the optimizers and the trainer.
- `nnet::data` adds `DataFrame` to Tensor conversion, for loading CSV files.
- `nnet::core` is just `Tensor` and the plain operations, without autograd.
- `nnet::csv` is just the `DataFrame` CSV reader, with no Tensor code at all.

Linking one brings in what it depends on, including the C++17 setting, the
header paths and the threads library matmul uses. The layers (`Dense`,
`Embedding`, `Sequential` and the rest) and checkpoints are header-only, so they
compile as part of your program. nnet's warning flags stay inside nnet and
don't get forced on your code.

Training goes almost twice as fast if everything is compiled for your own CPU,
by adding `-march=native`. The catch is that the programs then only run on CPUs
like yours.

## Other builds

Only the frozen reference model, without compiling any Tensor or autograd code:

```bash
cmake --build --preset debug --target reference_mlp reference_tests
ctest --preset debug -R '^reference$'
```

Only the libraries, with no tests or examples:

```bash
cmake -S . -B build/cmake-library -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DNNET_BUILD_EXAMPLES=OFF
cmake --build build/cmake-library
```

To check that an installed copy works from a separate project, there's a small
one in `tests/cmake/consumer` that runs a Dense layer forward, backward and
through an optimizer step:

```bash
cmake --install build/cmake-release --prefix "$PWD/build/cmake-install"
cmake -S tests/cmake/consumer -B build/cmake-consumer -DCMAKE_PREFIX_PATH="$PWD/build/cmake-install"
cmake --build build/cmake-consumer
ctest --test-dir build/cmake-consumer --output-on-failure
```

The examples read their data with relative paths, so run them from the
repository root. The original Makefile still works on its own as well.

## What's in the build

- Libraries: `nnet_core`, `nnet_framework`, `nnet_csv`, `nnet_data`. These get
  installed, and each has a `nnet::` name for other projects to use.
- `nnet_reference`: the frozen reference model. It's only used by the tests and
  `reference_mlp`, and isn't installed.
- Examples: `reference_mlp` and `tensor_mlp`. They're controlled by
  `NNET_BUILD_EXAMPLES`, which is on when nnet is built by itself and off when
  another project pulls it in.
- Tests: `reference_tests`, `tensor_tests`, `parity_tests` and
  `optimizer_tests`. Each runs in its own temporary folder, and none of them need
  downloaded data.

## What's been checked

On Fedora WSL with GCC 16: all four test programs in all four presets, the
examples, the reference-only and library-only builds, the consumer project
against a moved install folder, and a separate project pulling nnet in with
FetchContent from a local clone. There's no hosted CI or Clang run yet, and
nothing has been tried on other platforms.

CMake docs:
[targets and usage requirements](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html),
[presets](https://cmake.org/cmake/help/v3.20/manual/cmake-presets.7.html),
[FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html),
[installing and exporting](https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html).
