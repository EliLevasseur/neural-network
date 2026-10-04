#ifndef PARAMETER_H
#define PARAMETER_H

#include "nnet/core/tensor.h"
#include "nnet/core/autograd/value.h"
#include <utility>
#include <vector>
#include <utility>
#include <string>
#include <stdexcept>

namespace nnet {


	// A Parameter is one trainable thing in a model: a value together with
	// the gradient that belongs to it. Keeping them in one object means they
	// cannot drift apart or be mismatched by hand.
	//
	// It holds state and nothing else. It does not know how to run a forward
	// pass and it does not know how to update itself. Deciding how to update
	// is the optimizer's job, and doing so here would fuse the two.
	struct Parameter {
		Tensor value;
		Tensor grad;
		bool requiresGrad = true;

		// The gradient is born with the same shape as the value it belongs
		// to, and is zero-filled so it can be read before anything has
		// written to it.
		explicit Parameter(Tensor initialValue)
		: value(std::move(initialValue)),
		  grad(value.shape(), std::vector<double>(value.numel(), 0.0)),
          originalShape_(value.shape()) {}

        // Public tensors may be replaced, but their construction shape is fixed.
        // Reject invalid replacement at computation/update boundaries.
        void validateShape() const {
            if (value.shape() != originalShape_ || grad.shape() != originalShape_) {
                throw std::invalid_argument("Parameter value/grad shape changed");
            }
        }

		Parameter(const Parameter&) = delete;
		Parameter& operator=(const Parameter&) = delete;

		// Keep the Parameter at a stable address while an optimizer observes it.
		// Containers move owning pointers to modules, not their Parameters.
		Parameter(Parameter&&) = delete;
		Parameter& operator=(Parameter&&) = delete;

    private:
        const Tensor::Shape originalShape_;
	};

	struct NamedParameter {
		std::string name;
		Parameter* parameter;
	};

	// A leaf standing in for this Parameter during one forward pass. It holds a
	// copy of the current value, and points back at the Parameter's gradient so
	// backward can deposit into it once every gradient is complete.
	//
	// Takes a const reference so a const forward can call it. The gradient is
	// written later, by backward, never by the forward pass itself. Every
	// Parameter lives inside a non-const module, so writing through this
	// pointer is well defined.
	inline Value leafFor(const Parameter& parameter) {
		parameter.validateShape();
		Value leaf = makeLeaf(parameter.value);
		if (parameter.requiresGrad && gradientRecordingEnabled) {
			leaf->accumulateInto = const_cast<Tensor*>(&parameter.grad);
		}
		return leaf;
	}
}

#endif
