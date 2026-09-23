#ifndef PARAMETER_H
#define PARAMETER_H

#include "nnet/core/tensor.h"
#include <utility>
#include <vector>

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

		// The gradient is born with the same shape as the value it belongs
		// to, and is zero-filled so it can be read before anything has
		// written to it.
		explicit Parameter(Tensor initialValue)
		: value(std::move(initialValue)),
		  grad(value.shape(), std::vector<double>(value.numel(), 0.0)) {}

		// A Parameter is a thing, not a value. The optimizer has to update
		// the exact object the forward pass read, so copying one is almost
		// always a mistake. Deleting the copy turns the classic silent bug
		//
		//     for (auto parameter : parameters)   // copies each one
		//         parameter.value = ...;          // updates a temporary
		//
		// into a compile error instead of a model that never trains.
		Parameter(const Parameter&) = delete;
		Parameter& operator=(const Parameter&) = delete;

		// Declaring the copy operations above stops the compiler generating
		// these, and without them a Parameter cannot live in a std::vector.
		Parameter(Parameter&&) = default;
		Parameter& operator=(Parameter&&) = default;
	};
}

#endif
