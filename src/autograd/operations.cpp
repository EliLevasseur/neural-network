#include "nnet/core/autograd/operations.h"

namespace nnet {

    // ---- backward halves ---------------------------------------------------
    // Each one runs only during backward. It receives the gradient of its own
    // output, and adds a share into each input's gradient.

    void SigmoidRecord::sendGradBackward(const Tensor& gradientOfOutput) {
        Tensor contribution = gradientOfOutput;
        contribution.inplaceMultiplication(sigmoidDerivitive(inputs[0]->data));
        inputs[0]->grad = inputs[0]->grad + contribution;
    }

    void ReLURecord::sendGradBackward(const Tensor& gradientOfOutput) {
        Tensor contribution = gradientOfOutput;
        contribution.inplaceMultiplication(reluDerivitive(inputs[0]->data));
        inputs[0]->grad = inputs[0]->grad + contribution;
    }

    void matmulRecord::sendGradBackward(const Tensor& gradientOfOutput) {
        // one gradient for each input
        inputs[0]->grad = inputs[0]->grad + matmulGradInput(gradientOfOutput, inputs[1]->data);
        // A weight shared by every group in a batched input collects one
        // contribution per group, so fold them back to the weight's own shape.
        inputs[1]->grad = inputs[1]->grad + sumLeadingDimensions(
            matmulGradWeight(inputs[0]->data, gradientOfOutput), inputs[1]->data.rank());
    }

    void BiasAddRecord::sendGradBackward(const Tensor& gradientOfOutput) {
        inputs[0]->grad = inputs[0]->grad + gradientOfOutput;
        inputs[1]->grad = inputs[1]->grad + addBiasGrad(gradientOfOutput, inputs[1]->data);
    }

    void AddRecord::sendGradBackward(const Tensor& gradientOfOutput) {
        inputs[0]->grad = inputs[0]->grad + gradientOfOutput;
        inputs[1]->grad = inputs[1]->grad + gradientOfOutput;
    }

    // Only the prediction gets a gradient. The target is data, so nothing
    // upstream of it can learn.
    void LossRecord::sendGradBackward(const Tensor& gradientOfOutput) {
        Tensor contribution = binaryCrossEntropyGrad(inputs[0]->data, inputs[1]->data);
        contribution = contribution * gradientOfOutput.at({0});
        inputs[0]->grad = inputs[0]->grad + contribution;
    }

    // Same shape as LossRecord: only the scores get a gradient.
    void SoftmaxCrossEntropyRecord::sendGradBackward(const Tensor& gradientOfOutput) {
        Tensor contribution = softmaxCrossEntropyGrad(inputs[0]->data, inputs[1]->data);
        contribution = contribution * gradientOfOutput.at({0});
        inputs[0]->grad = inputs[0]->grad + contribution;
    }

    // ---- forward halves ----------------------------------------------------
    // Each one computes its output with the plain Tensor operation, and leaves
    // a record behind for backward. makeOutput drops the record under NoGrad.

    Value sigmoid(const Value& input) {
        auto record = std::make_shared<SigmoidRecord>();
        record->inputs = {input};
        return makeOutput(sigmoid(input->data), record);
    }

    Value relu(const Value& input) {
        auto record = std::make_shared<ReLURecord>();
        record->inputs = {input};
        return makeOutput(relu(input->data), record);
    }

    Value add(const Value& left, const Value& right) {
        auto record = std::make_shared<AddRecord>();
        record->inputs = {left, right};
        return makeOutput(left->data + right->data, record);
    }

    Value matmul(const Value& left, const Value& right) {
        auto record = std::make_shared<matmulRecord>();
        record->inputs = {left, right};
        return makeOutput(left->data * right->data, record);
    }

    Value addBias(const Value& input, const Value& bias) {
        auto record = std::make_shared<BiasAddRecord>();
        record->inputs = {input, bias};

        // Copy first: the Tensor addBias works in place, and the record still
        // needs the input's original numbers.
        Tensor output = input->data;
        addBias(output, bias->data);
        return makeOutput(std::move(output), record);
    }

    Value binaryCrossEntropy(const Value& prediction, const Value& target) {
        auto record = std::make_shared<LossRecord>();
        record->inputs = {prediction, target};

        // A single number, carried as shape {1}.
        const double loss = binaryCrossEntropy(prediction->data, target->data);
        return makeOutput(Tensor({1}, {loss}), record);
    }

    Value softmaxCrossEntropy(const Value& scores, const Value& target) {
        auto record = std::make_shared<SoftmaxCrossEntropyRecord>();
        record->inputs = {scores, target};

        // A single number, carried as shape {1}.
        const double loss = softmaxCrossEntropy(scores->data, target->data);
        return makeOutput(Tensor({1}, {loss}), record);
    }
}
