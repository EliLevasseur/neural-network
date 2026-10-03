#ifndef MODULE_H
#define MODULE_H

#include <vector>
#include "parameter.h"
#include "../core/tensor_ops.h"

namespace nnet {
	class Module {
		public:
			virtual ~Module();
			
			virtual std::vector<Parameter*> parameters(){
				return {};
		};
	};

	class Unary_Module : public Module {
		public:
			virtual ~Unary_Module() = default;
			virtual Tensor Operation(Tensor&) {
				return Tensor{{1, 1}, {0}};
			}
			
			virtual std::vector<Parameter*> parameters(){
				return {};
			}
	};

	class Sigmoid : public Unary_Module {
        Tensor Operation(Tensor& tensor) override {
            return sigmoid(tensor);
        }
           
    };

}




#endif
