#ifndef BACKWARD_H
#define BACKWARD_H

#include "value.h"

namespace nnet {
    // Starts from a gradient of ones, which is right when `loss` is the
    // number being minimised.
    void backward(const Value& loss);

    // Starts from a gradient you supply instead. Useful when `output` is not
    // a loss, for example to check one layer against a known incoming
    // gradient. It must have the same shape as `output`.
    void backward(const Value& output, const Tensor& gradientOfOutput);
}

#endif