#ifndef VALUE_H
#define VALUE_H

#include <memory>
#include <vector>

#include "nnet/core/tensor.h"
#include "nnet/core/tensor_ops.h"

namespace nnet {

    struct OperationRecord;

    struct ValueState {
        Tensor data;
        Tensor grad;
        std::shared_ptr<OperationRecord> madeBy;

        // for storing the parameters gradients within the model, so we can update them with an optimizer
        Tensor* accumulateInto = nullptr;

    };

    using Value = std::shared_ptr<ValueState>;


    struct OperationRecord {
        std::vector<Value> inputs;

        // Set once backward has used this record. Its inputs are released at
        // the same time, so a second backward over the same graph is refused
        // instead of quietly depositing every gradient a second time.
        bool consumed = false;

        virtual void sendGradBackward(const Tensor& gradientOfOutput) = 0;
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
    inline Value makeOutput(Tensor data, std::shared_ptr<OperationRecord> record) {
        Tensor zeroGrad = zeros(data.shape());
        if (!gradientRecordingEnabled) {
            record = nullptr;
        }
        return std::make_shared<ValueState>(
            ValueState{std::move(data), std::move(zeroGrad), std::move(record)});
    }

    // A leaf is a value no operation produced, so nothing made it.
    inline Value makeLeaf(Tensor data) {
        Tensor zeroGrad = zeros(data.shape());
        return std::make_shared<ValueState>(ValueState{std::move(data), std::move(zeroGrad), nullptr});
    }

}
#endif