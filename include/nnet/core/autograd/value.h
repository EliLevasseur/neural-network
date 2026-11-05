#ifndef VALUE_H
#define VALUE_H

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "nnet/core/tensor.h"
#include "nnet/core/tensor_ops.h"

namespace nnet {

    struct OperationRecord;

    struct ValueState {
        Tensor data;
        // Empty until backward sends this value its first gradient. Values
        // that never get one (data, frozen weights, anything under NoGrad)
        // never pay for a zero-filled tensor.
        std::optional<Tensor> grad;
        std::shared_ptr<OperationRecord> madeBy;

        // for storing the parameters gradients within the model, so we can update them with an optimizer
        Tensor* accumulateInto = nullptr;
        std::uint64_t visitedPass = 0;

        // Whether backward should compute a gradient for this value. True for
        // anything that depends on a trainable value; false for plain data.
        bool requiresGrad = false;

    };

    using Value = std::shared_ptr<ValueState>;


    struct OperationRecord {
        std::vector<Value> inputs;

        // Set once backward has used this record. Its inputs are released at
        // the same time, so a second backward over the same graph is refused
        // instead of quietly depositing every gradient a second time.
        bool consumed = false;

        // `output` is the value this operation produced in the forward pass,
        // for operations whose derivative is easiest to write in terms of it.
        virtual void sendGradBackward(const Tensor& gradientOfOutput, const Tensor& output) = 0;
        virtual ~OperationRecord() = default;
    };

    // Operations only leave records while this is true. NoGrad switches it
    // off for a block of code, for work like measuring the loss, where
    // nothing will ever call backward.
    inline thread_local bool gradientRecordingEnabled = true;

    // Turns recording off for as long as it exists, then restores whatever
    // the setting was before:

    class NoGrad {
        public:
            NoGrad() : previous_(gradientRecordingEnabled) {
                gradientRecordingEnabled = false;
            }
            ~NoGrad() { gradientRecordingEnabled = previous_; }

            NoGrad(const NoGrad&) = delete;
            NoGrad& operator=(const NoGrad&) = delete;

        private:
            bool previous_;
    };

    // Builds the Value an operation returns. Every operation goes through
    // here, so recording is switched on or off in exactly one place.
    //
    // The output needs a gradient only if one of its inputs does. If none do,
    // the record is dropped too: backward will never need it.
    inline Value makeOutput(Tensor data, std::shared_ptr<OperationRecord> record) {
        bool needsGrad = false;
        if (gradientRecordingEnabled) {
            for (const Value& input : record->inputs) {
                if (input->requiresGrad) {
                    needsGrad = true;
                    break;
                }
            }
        }
        if (!needsGrad) {
            record = nullptr;
        }
        Value output = std::make_shared<ValueState>(
            ValueState{std::move(data), std::nullopt, std::move(record)});
        output->requiresGrad = needsGrad;
        return output;
    }

    // A leaf is a value no operation produced, so nothing made it. Pass
    // false for plain data (inputs, targets), so backward skips it.
    inline Value makeLeaf(Tensor data, bool requiresGrad = true) {
        Value leaf = std::make_shared<ValueState>(ValueState{std::move(data), std::nullopt, nullptr});
        leaf->requiresGrad = requiresGrad;
        return leaf;
    }

    // Adds a contribution to a value's gradient. The first one is moved in
    // as the gradient itself, so nothing is zero-filled or copied.
    inline void accumulateGrad(ValueState& value, Tensor contribution) {
        if (!value.requiresGrad) {
            return;
        }
        if (value.grad) {
            *value.grad += contribution;
        } else {
            value.grad = std::move(contribution);
        }
    }

    // The gradient to add into element by element, created as zeros the
    // first time. For backward loops that write straight into it.
    inline Tensor& gradToAddInto(ValueState& value) {
        if (!value.grad) {
            value.grad = zeros(value.data.shape());
        }
        return *value.grad;
    }

}
#endif