#include "nnet/core/autograd/backward.h"

#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace nnet {

    // Adds a value to `order` only after everything it was made from.
    static void collectInOrder(const Value& value, std::vector<ValueState*>& order,
                               std::unordered_set<ValueState*>& visited) {
        if (!visited.insert(value.get()).second) return;
        if (value->madeBy) {
            for (const Value& input : value->madeBy->inputs) {
                collectInOrder(input, order, visited);
            }
        }
        order.push_back(value.get());
    }

    void backward(const Value& loss) {
        backward(loss, fill(loss->data.shape(), 1.0));
    }

    void backward(const Value& output, const Tensor& gradientOfOutput) {
        if (gradientOfOutput.shape() != output->data.shape()) {
            throw std::invalid_argument(
                "backward: the starting gradient must match the output's shape");
        }

        std::vector<ValueState*> order;
        std::unordered_set<ValueState*> visited;
        collectInOrder(output, order, visited);

        // Refuse a used graph before touching anything, so a rejected call
        // leaves every gradient exactly as it was.
        for (ValueState* value : order) {
            if (value->madeBy && value->madeBy->consumed) {
                throw std::logic_error(
                    "backward has already run on this graph; build a new one with a fresh forward pass");
            }
        }

        output->grad = gradientOfOutput;

        for (auto it = order.rbegin(); it != order.rend(); ++it) {
            if ((*it)->madeBy) {
                (*it)->madeBy->sendGradBackward((*it)->grad);
            }
        }

        // Every gradient is now complete, so leaves standing in for something
        // persistent hand theirs over. It is a transfer, not a copy: the leaf
        // resets to zero afterwards. Otherwise a leaf reused by a later graph
        // would carry this gradient forward and deposit it a second time.
        for (ValueState* value : order) {
            if (value->accumulateInto) {
                *value->accumulateInto = *value->accumulateInto + value->grad;
                value->grad = zeros(value->grad.shape());
            }
        }

        // Release the graph. The records are collected first and held here,
        // because clearing one record's inputs can free values that `order`
        // still points at. After this loop nothing touches `order` again.
        std::vector<std::shared_ptr<OperationRecord>> records;
        for (ValueState* value : order) {
            if (value->madeBy) {
                records.push_back(value->madeBy);
            }
        }
        for (const auto& record : records) {
            record->consumed = true;
            record->inputs.clear();
        }
    }
}
