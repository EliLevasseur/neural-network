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

// Swaps each id for its table row (see embeddingLookup). Only the table
// gets a gradient; the ids are row numbers, so they never get one.
Value embedding(const Value& ids, const Value& table);

struct AddRecord : OperationRecord {
    void sendGradBackward(const Tensor& gradientOfOutput, const Tensor& output) override;
};
struct matmulRecord : OperationRecord {
    void sendGradBackward(const Tensor& gradientOfOutput, const Tensor& output) override;
};
struct SigmoidRecord : OperationRecord {
    void sendGradBackward(const Tensor& gradientOfOutput, const Tensor& output) override;
};

struct ReLURecord : OperationRecord {
    void sendGradBackward(const Tensor& gradientOfOutput, const Tensor& output) override;
};

struct BiasAddRecord : OperationRecord {
    void sendGradBackward(const Tensor& gradientOfOutput, const Tensor& output) override;
};

struct LossRecord : OperationRecord {
    void sendGradBackward(const Tensor& gradientOfOutput, const Tensor& output) override;
};

struct SoftmaxCrossEntropyRecord : OperationRecord {
    void sendGradBackward(const Tensor& gradientOfOutput, const Tensor& output) override;
};

struct EmbeddingRecord : OperationRecord {
    void sendGradBackward(const Tensor& gradientOfOutput, const Tensor& output) override;
};

}

#endif
