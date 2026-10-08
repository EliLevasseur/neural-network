#ifndef NNET_DATA_MNIST_H
#define NNET_DATA_MNIST_H

#include "nnet/core/tensor.h"

#include <string>

// Reads the MNIST handwritten-digit files, in the IDX format they are
// published in, straight into the two Tensors the trainer takes.
//
// An IDX file is a short header followed by raw bytes. The header starts with
// a "magic number" saying what kind of file it is (2051 for images, 2049 for
// labels), then the number of examples, and for images the height and width.
// Header numbers are stored big-endian: most significant byte first.

namespace nnet {

    struct MnistData {
        // [count, 784]: each 28x28 image flattened row by row, with every
        // pixel scaled from 0..255 down to 0..1.
        Tensor images;
        // [count, 10]: one-hot rows, a 1 in the column of the correct digit.
        Tensor labels;
    };

    // Throws std::runtime_error if a file is missing, is not the expected
    // kind, is not exactly the size its header describes, holds images that
    // are not 28x28 or a label above 9, or if the two files disagree on how
    // many examples there are.
    MnistData loadMnist(const std::string& imagePath, const std::string& labelPath);

}

#endif
