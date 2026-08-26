#include <vector>
#include <stdexcept>

namespace nnet {

class Tensor {
	

	public:
		using Shape = std::vector<std::size_t>;	

		Tensor(Shape shape, std::vector<double> values);
		std::size_t rank() const;
		std::size_t numel() const;
		
		const Shape& strides() const;		
	private:
		Shape shape_;
		Shape strides_;
		std::vector<double> data_;
		
	};
}
