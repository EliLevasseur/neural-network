#ifndef MODULE_H
#define MODULE_H

#include <algorithm>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>
#include "parameter.h"
#include "nnet/core/autograd/operations.h"

namespace nnet {
	class Module;

	// Names are owned strings; module pointers observe children owned elsewhere.
	struct NamedModule {
		std::string name;
		Module* module;
	};

	class Module {
		public:

			Module() = default;

			// A module cannot be copied or moved; owning pointers may move.
			Module(const Module&) = delete;
			Module& operator=(const Module&) = delete;

			Module(Module&&) = delete;
			Module& operator=(Module&&) = delete;

			virtual ~Module() = default;
			
			std::vector<NamedParameter> namedParameters() {
				std::vector<NamedParameter> result;
				std::unordered_set<Module*> visitedModules;
				std::unordered_set<Parameter*> visitedParameters;
				collectNamedParameters("", result, visitedModules, visitedParameters);
				return result;
			}

			// The optimizer-facing view uses the same discovery and ordering.
			std::vector<Parameter*> parameters() {
				std::vector<Parameter*> result;
				for (const auto& entry : namedParameters()) {
					result.push_back(entry.parameter);
				}
				return result;
			}

			// Clears the gradients in the models components.
			//
			// The parameter list is found once and kept: a module's children
			// are fixed when it's built, so the list can't change, and
			// rebuilding it (names, hash sets) every step was most of the cost.
			void zeroGrad() {
                if (!parametersCached_) {
                    cachedParameters_ = parameters();
                    parametersCached_ = true;
                }
                for (const Parameter* param : cachedParameters_) {
                    param->validateShape();
                }
                for (Parameter* param : cachedParameters_) {
                    double* grad = param->grad.data();
                    std::fill(grad, grad + param->grad.numel(), 0.0);
                }
			}

		protected:
		
			virtual std::vector<NamedParameter> localNamedParameters() {
				return {};
			}

			virtual std::vector<NamedModule> namedChildren() {
				return {};
			}

		private:
			std::vector<Parameter*> cachedParameters_;
			bool parametersCached_ = false;

			static void validateLocalName(const std::string& name,
			                              std::unordered_set<std::string>& names) {
				if (name.empty() || name.find(".") != std::string::npos ||
				    !names.insert(name).second) {
					throw std::invalid_argument("Invalid or duplicate local name: " + name);
				}
			}

			void collectNamedParameters(const std::string& prefix,
			                            std::vector<NamedParameter>& result,
			                            std::unordered_set<Module*>& visitedModules,
			                            std::unordered_set<Parameter*>& visitedParameters) {
				if (!visitedModules.insert(this).second) {
					throw std::invalid_argument("Module discovery requires a unique ownership tree");
				}

				std::unordered_set<std::string> parameterNames;
				for (const auto& entry : localNamedParameters()) {
					validateLocalName(entry.name, parameterNames);
					if (!entry.parameter || !visitedParameters.insert(entry.parameter).second) {
						throw std::invalid_argument("Null or repeated Parameter in discovery");
					}
					result.push_back({prefix + entry.name, entry.parameter});
				}

				std::unordered_set<std::string> childNames;
				for (const auto& child : namedChildren()) {
					validateLocalName(child.name, childNames);
					if (!child.module) {
						throw std::invalid_argument("Null child Module in discovery");
					}
					child.module->collectNamedParameters(prefix + child.name + ".",
					    result, visitedModules, visitedParameters);
				}
			}

		};

	class Unary_Module : public Module {
		public:
            // Common public entry; derived components supply only the operation.
            Value forward(const Value& input) const {
                return forward_impl(input);
            }

        protected:
            virtual Value forward_impl(const Value& input) const = 0;

	};

	class Sigmoid : public Unary_Module {
		protected:
			Value forward_impl(const Value& input) const override {
				return sigmoid(input);
			}
           
    };

	class ReLU : public Unary_Module {
		protected:
			Value forward_impl(const Value& input) const override {
				return relu(input);
			}
		   
	};

}

#endif
