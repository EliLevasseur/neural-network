#ifndef SEQUENTIAL_H
#define SEQUENTIAL_H

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "module.h"


namespace nnet {
    // The representation of a sequential forward pass.
    // This class takes ownership of the forward pass and ensures the Unary_Modules Tensor in Tensor out contract.
    class Sequential : public Unary_Module {
        public:

            Sequential(std::vector<std::unique_ptr<Unary_Module>> components) : components_{std::move(components)} {}

        protected:
            Value forward_impl(const Value& input) const override {
                Value output = input;
                for (auto& component : components_) {
                    output = component->forward(output);
                }
                return output;
            }

            // Every component keeps its index, including parameter-free activations.
            // Module adds these prefixes while recursively discovering parameters.
            std::vector<NamedModule> namedChildren() override {
                std::vector<NamedModule> result;
                result.reserve(components_.size());
                for (std::size_t index = 0; index < components_.size(); ++index) {
                    result.push_back({std::to_string(index), components_[index].get()});
                }
                return result;
            }

        private:
            std::vector<std::unique_ptr<Unary_Module>> components_;
    };
}



#endif