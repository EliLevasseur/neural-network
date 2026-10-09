# Building with CMake

Run these commands from the repository root inside FedoraLinux-44 WSL.
CMake 3.20+, GNU Make, and GCC with C++17 support are sufficient. The presets
select `g++`; they do not require Ninja or download dependencies.

## Build and test

```bash
cmake --preset debug
cmake --build --preset debug
ctest --preset debug
```

CMake configures targets, the build command compiles/links them, and CTest runs
the existing test programs. Four CTest entries contain 411 individual checks:
reference (36), Tensor/framework (225), parity (104), and optimizer (46).

| Preset | Purpose | Directory |
|---|---|---|
| debug | GCC, -g3 -O0 | build/cmake-debug |
| checked | Debug with checked libstdc++ containers | build/cmake-checked |
| release | GCC, -O2 -DNDEBUG | build/cmake-release |
| asan | Debug with AddressSanitizer and UndefinedBehaviorSanitizer | build/cmake-asan |

Replace `debug` with another name in all three commands. Sanitizers require
the compiler's ASan/UBSan runtime libraries. Presets build with two jobs; override
with `cmake --build --preset debug --parallel 4` if desired.

The checked preset changes the standard-library ABI. Its definitions propagate
through library targets so consumers use the same layout. Sanitizer runtime link
options propagate too. Use separate build directories for different modes.

Each build contains `compile_commands.json` for editor tooling. Optional local
overrides belong in the ignored `CMakeUserPresets.json`.

## Targets and dependencies

| Target | Purpose / dependency |
|---|---|
| nnet_core / nnet::core | Tensor and numerical operations |
| nnet_framework / nnet::framework | Autograd, optimizers and trainer; uses core |
| nnet_csv / nnet::csv | Tensor-independent DataFrame CSV reader |
| nnet_data / nnet::data | Tensor/DataFrame adapter and MNIST loader; uses core and CSV |
| nnet_reference | Frozen Network/Trainer oracle; not installed |
| reference_mlp | Reference executable; uses reference and CSV only |
| tensor_mlp, mnist | Framework examples; use framework and data |
| reference_tests, tensor_tests, parity_tests, optimizer_tests | Existing suites |

Parameter, Module, Dense, Sequential and checkpoint implementations live in
headers. They are installed with the framework and compile in consumers.

The `PUBLIC` requirements pass headers, C++17, and necessary dependency/ABI
settings to consumers. Warnings are `PRIVATE`, so your warning policy is not
imposed on another project. Sources are listed explicitly to preserve boundaries.

To build only the reference oracle, without compiling Tensor or autograd:

```bash
cmake --build --preset debug --target reference_mlp reference_tests
ctest --preset debug -R '^reference$'
```

CTest copies the small CSV fixture into the build directory and assigns a
separate temporary directory per suite/build. Tests do not need downloaded MNIST
or the shell's working directory to be the source tree.

## Run examples

Examples read relative dataset paths, so run them from the repository root.
Use Release for full training; CMake's Debug MNIST build stays unoptimized.

```bash
cmake --preset release
cmake --build --preset release
./build/cmake-release/reference_mlp
./build/cmake-release/tensor_mlp
make mnist-data
./build/cmake-release/mnist
```

Downloading MNIST remains an explicit existing Make target. Configuring,
compiling, and testing never download or train on the full dataset.
The original Makefile and its commands still work independently.

## Install and use from another project

```bash
cmake --preset release
cmake --build --preset release
cmake --install build/cmake-release --prefix "$PWD/build/cmake-install"

cmake -S tests/cmake/consumer -B build/cmake-consumer \
  -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_PREFIX_PATH="$PWD/build/cmake-install"
cmake --build build/cmake-consumer
ctest --test-dir build/cmake-consumer --output-on-failure
```

The consumer is an independent CMake project. It uses only:

```cmake
find_package(nnet CONFIG REQUIRED)
target_link_libraries(your_program PRIVATE nnet::framework)
```

Link `nnet::data` as well when using the loaders. CMake supplies transitive
libraries and installed header paths. The install tree can be relocated;
no source-checkout path is exported. This is local package support, not a
published release or a cross-platform binary compatibility promise.

For a library-only build:

```bash
cmake -S . -B build/cmake-library -DCMAKE_CXX_COMPILER=g++ \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF -DNNET_BUILD_EXAMPLES=OFF
cmake --build build/cmake-library
```

## Verification scope

Verified on FedoraLinux-44 WSL with GCC: all four suites in all four presets,
all example builds, a reference-only build, a library-only build, and a copied
consumer project using a relocated installation. The consumer checks a Dense
forward, autograd backward, and optimizer update through installed headers/libs.

Hosted CI and a Clang run are still M6 work. No other platform has been verified.

CMake references: [targets and usage requirements](https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html),
[presets](https://cmake.org/cmake/help/v3.20/manual/cmake-presets.7.html),
[installation and exports](https://cmake.org/cmake/help/latest/guide/importing-exporting/index.html).
