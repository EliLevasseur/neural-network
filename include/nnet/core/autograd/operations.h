#ifndef OPERATIONS_H
#define OPERATIONS_H

#include "nnet/core/autograd/value.h"
#include "nnet/core/tensor_ops.h"

namespace nnet {

// Activation functions
Value sigmoid(const Value& input);
Value relu(const Value& input);

// Helper functions for layers
Value matmul(const Value& input, const Value& weight);
Value add(const Value& left, const Value& right);
Value addBias(const Value& input, const Value& bias);

// Loss functions
Value binaryCrossEntropy(const Value& prediction, const Value& target);
Value softmaxCrossEntropy(const Value& scores, const Value& target);

struct AddRecord : OperationRecord {
    void sendGradBackward(const Tensor& gradientOfOutput) override;
};
struct matmulRecord : OperationRecord {
    void sendGradBackward(const Tensor& gradientOfOutput) override ;
};
struct SigmoidRecord : OperationRecord {
    void sendGradBackward(const Tensor& gradientOfOutput) override;
};

struct ReLURecord : OperationRecord {
    void sendGradBackward(const Tensor& gradientOfOutput) override;
};

struct BiasAddRecord : OperationRecord {
    void sendGradBackward(const Tensor& gradientOfOutput) override;
};

struct LossRecord : OperationRecord {
    void sendGradBackward(const Tensor& gradientOfOutput) override;
};

struct SoftmaxCrossEntropyRecord : OperationRecord {
    void sendGradBackward(const Tensor& gradientOfOutput) override;
};

}

#endif
