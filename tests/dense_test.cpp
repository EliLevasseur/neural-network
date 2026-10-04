#include "test_utils.h"
#include "nnet/nn/dense.h"
#include "nnet/nn/sequential.h"
#include "nnet/core/autograd/backward.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <type_traits>
#include <vector>

static_assert(!std::is_copy_constructible_v<nnet::Module> &&
              !std::is_copy_assignable_v<nnet::Module> &&
              !std::is_move_constructible_v<nnet::Module> &&
              !std::is_move_assignable_v<nnet::Module>,
              "Modules must keep stable object identity");
static_assert(!std::is_copy_constructible_v<nnet::Parameter> &&
              !std::is_copy_assignable_v<nnet::Parameter> &&
              !std::is_move_constructible_v<nnet::Parameter> &&
              !std::is_move_assignable_v<nnet::Parameter>,
              "Parameters must keep stable object identity");
static_assert(!std::is_copy_constructible_v<nnet::Dense> &&
              !std::is_copy_assignable_v<nnet::Dense> &&
              !std::is_move_constructible_v<nnet::Dense> &&
              !std::is_move_assignable_v<nnet::Dense>,
              "Dense components must inherit the stable identity contract");

namespace {
    // Runs a module on plain numbers and returns plain numbers.
    nnet::Tensor forwardOf(const nnet::Unary_Module& module, const nnet::Tensor& input) {
        return module.forward(nnet::makeLeaf(input))->data;
    }

    // The autograd version of the old Dense::backward(input, upstream): build
    // the graph, start backward from that exact incoming gradient, and hand
    // back the gradient that reached the input. The layer's own gradients land
    // in its Parameters along the way.
    nnet::Tensor backwardThrough(const nnet::Unary_Module& module,
                                 const nnet::Tensor& input,
                                 const nnet::Tensor& upstream) {
        nnet::Value inputLeaf = nnet::makeLeaf(input);
        nnet::backward(module.forward(inputLeaf), upstream);
        return inputLeaf->grad;
    }
}

void runDenseTests(TestRunner& tests) {
	tests.section("DENSE LAYER TESTS");

	// ---- construction ---------------------------------------------------
	// Both shapes come from the same two numbers, so a weight sized for one
	// layer can never sit next to a bias sized for another.
	nnet::Dense layer(3, 2);

	tests.expectTrue(layer.weight.value.shape() == nnet::Tensor::Shape{3, 2},
		"Dense(3, 2) creates a weight shaped inputs by outputs");

	tests.expectTrue(layer.bias.value.shape() == nnet::Tensor::Shape{2},
		"Dense(3, 2) creates one bias value per output");

	// A gradient is born matching the value it belongs to, and starts at
	// zero so it can be read before anything has written to it.
	tests.expectTrue(layer.weight.grad.shape() == layer.weight.value.shape() &&
		layer.bias.grad.shape() == layer.bias.value.shape(),
		"each gradient is born with the same shape as its value");

	tests.expectTrue(layer.weight.grad.getData() == std::vector<double>(6, 0.0) &&
		layer.bias.grad.getData() == std::vector<double>(2, 0.0),
		"gradients start filled with zero");

	// ---- parameters() ---------------------------------------------------
	const std::vector<nnet::Parameter*> collected = layer.parameters();

	tests.expectTrue(collected.size() == 2,
		"parameters() reports both trainable values of the layer");

	tests.expectTrue(collected[0] == &layer.weight && collected[1] == &layer.bias,
		"parameters() points at the layer's own members, not at copies");

	// The point of handing out pointers is that an optimizer can change the
	// real weights. Writing through the pointer must be visible in the layer.
	collected[0]->value.at({0, 0}) = 99.0;
	tests.expectNear(layer.weight.value.at({0, 0}), 99.0, 1.0e-12,
		"writing through a parameter pointer changes the layer itself");

	// ---- stable ownership -----------------------------------------------
	// Force the owning vector to reallocate after parameter discovery.
	// The pointers move, but the Dense and its Parameters must stay put.
	std::vector<std::unique_ptr<nnet::Dense>> ownedLayers;
	ownedLayers.push_back(std::make_unique<nnet::Dense>(2, 2));
	nnet::Dense* originalDense = ownedLayers.front().get();
	const auto stableParameters = ownedLayers.front()->parameters();
	ownedLayers.reserve(ownedLayers.capacity() + 1);

	tests.expectTrue(ownedLayers.front().get() == originalDense &&
		stableParameters[0] == &ownedLayers.front()->weight &&
		stableParameters[1] == &ownedLayers.front()->bias,
		"owning-vector reallocation preserves Dense and Parameter addresses");

	stableParameters[0]->value.at({0, 0}) = 0.75;
	stableParameters[1]->value.at({0}) = -0.25;
	tests.expectTrue(ownedLayers.front()->weight.value.at({0, 0}) == 0.75 &&
		ownedLayers.front()->bias.value.at({0}) == -0.25,
		"saved parameter pointers still update the owned model after reallocation");

	// ---- forward --------------------------------------------------------
	// Weights are randomly initialised, so overwrite them with known values
	// to get a result that can be checked by hand.
	nnet::Dense knownLayer(2, 2);
	knownLayer.weight.value = nnet::Tensor({2, 2}, {1, 2,
	                                                3, 4});
	knownLayer.bias.value = nnet::Tensor({2}, {10, 20});

	const nnet::Tensor input({1, 2}, {1, 1});
	const nnet::Tensor output = forwardOf(knownLayer, input);

	// First output  = 1*1 + 1*3 = 4, plus bias 10 = 14
	// Second output = 1*2 + 1*4 = 6, plus bias 20 = 26
	tests.expectTrue(output.shape() == nnet::Tensor::Shape{1, 2},
		"forward returns one row with one value per output");

	tests.expectNear(output.at({0, 0}), 14.0, 1.0e-12,
		"forward computes the first output by hand-checked arithmetic");

	tests.expectNear(output.at({0, 1}), 26.0, 1.0e-12,
		"forward computes the second output by hand-checked arithmetic");

	// ---- forward keeps no state ------------------------------------------
	// A layer that cached its last input would return something different
	// the second time, and would compute a wrong gradient when reused.
	const nnet::Tensor secondOutput = forwardOf(knownLayer, input);
	tests.expectTrue(secondOutput.getData() == output.getData(),
		"calling forward twice with the same input gives the same answer");

	const nnet::Tensor otherInput({1, 2}, {2, 0});
	const nnet::Tensor otherOutput = forwardOf(knownLayer, otherInput);
	tests.expectNear(otherOutput.at({0, 0}), 12.0, 1.0e-12,
		"a later call is unaffected by an earlier one");

	tests.expectTrue(knownLayer.weight.value.at({0, 0}) == 1.0 &&
		knownLayer.bias.value.at({0}) == 10.0,
		"forward leaves the layer's own weight and bias untouched");
	// ---- a layer applied across grouped (rank-3) data -------------------
	// 2 groups, 2 rows each, 3 numbers per row, through one Dense(3, 2).
	nnet::Dense groupedLayer(3, 2);
	groupedLayer.weight.value = nnet::Tensor({3, 2}, {1, 0,
	                                                  0, 1,
	                                                  1, 1});
	groupedLayer.bias.value = nnet::Tensor({2}, {0, 0});

	const nnet::Tensor groupedInput({2, 2, 3}, {1, 2, 3,  4, 5, 6,
	                                            7, 8, 9, 10, 11, 12});
	const nnet::Tensor groupedOutput = forwardOf(groupedLayer, groupedInput);

	tests.expectTrue(groupedOutput.shape() == nnet::Tensor::Shape{2, 2, 2} &&
		groupedOutput.getData() == std::vector<double>{4, 5, 10, 11, 16, 17, 22, 23},
		"one layer runs across grouped data using a single shared weight");

	// Backward must fold every group's contribution back into the one weight,
	// so the gradient comes out shaped like the weight rather than the batch.
	const nnet::Tensor groupedUpstream({2, 2, 2}, {1, 1, 1, 1, 1, 1, 1, 1});
	backwardThrough(groupedLayer, groupedInput, groupedUpstream);

	tests.expectTrue(groupedLayer.weight.grad.shape() == nnet::Tensor::Shape{3, 2},
		"a shared weight's gradient has the weight's shape, not the batch's");

	// Independent check: do each group separately and add the two results.
	const nnet::Tensor firstGroup({2, 3}, {1, 2, 3, 4, 5, 6});
	const nnet::Tensor secondGroup({2, 3}, {7, 8, 9, 10, 11, 12});
	const nnet::Tensor onesRank2({2, 2}, {1, 1, 1, 1});
	const nnet::Tensor byHand =
		nnet::matmulGradWeight(firstGroup, onesRank2) +
		nnet::matmulGradWeight(secondGroup, onesRank2);

	tests.expectTrue(groupedLayer.weight.grad.getData() == byHand.getData(),
		"the grouped weight gradient equals each group done separately and added");

	tests.expectTrue(groupedLayer.bias.grad.shape() == nnet::Tensor::Shape{2} &&
		groupedLayer.bias.grad.getData() == std::vector<double>{4, 4},
		"the shared bias gradient sums over every group and row");

	// ---- gradient accumulation and clearing -----------------------------
	// For input 2 and incoming gradient 3: dW = 2 * 3 = 6, db = 3.
	// With weight 0.5, each call returns dInput = 3 * 0.5 = 1.5.
	nnet::Dense accumulatingLayer(1, 1);
	accumulatingLayer.weight.value = nnet::Tensor({1, 1}, {0.5});
	accumulatingLayer.bias.value = nnet::Tensor({1}, {0.25});
	const nnet::Tensor accumulationInput({1, 1}, {2.0});
	const nnet::Tensor accumulationUpstream({1, 1}, {3.0});

	// Begin with nonzero gradients so a no-op zeroGrad cannot pass.
	accumulatingLayer.weight.grad.at({0, 0}) = 7.0;
	accumulatingLayer.bias.grad.at({0}) = -4.0;
	accumulatingLayer.zeroGrad();
	tests.expectTrue(accumulatingLayer.weight.grad.at({0, 0}) == 0.0 &&
		accumulatingLayer.bias.grad.at({0}) == 0.0,
		"zeroGrad clears both nonzero parameter gradients");

	const nnet::Tensor firstInputGradient =
		backwardThrough(accumulatingLayer, accumulationInput, accumulationUpstream);
	tests.expectNear(accumulatingLayer.weight.grad.at({0, 0}), 6.0, 1.0e-12,
		"first backward adds the weight contribution");
	tests.expectNear(accumulatingLayer.bias.grad.at({0}), 3.0, 1.0e-12,
		"first backward adds the bias contribution");
	tests.expectNear(firstInputGradient.at({0, 0}), 1.5, 1.0e-12,
		"first backward returns its own input gradient");

	const nnet::Tensor secondInputGradient =
		backwardThrough(accumulatingLayer, accumulationInput, accumulationUpstream);
	tests.expectNear(accumulatingLayer.weight.grad.at({0, 0}), 12.0, 1.0e-12,
		"second backward accumulates rather than overwrites the weight gradient");
	tests.expectNear(accumulatingLayer.bias.grad.at({0}), 6.0, 1.0e-12,
		"second backward accumulates rather than overwrites the bias gradient");
	tests.expectNear(secondInputGradient.at({0, 0}), 1.5, 1.0e-12,
		"returned input gradient does not accumulate across calls");

	accumulatingLayer.zeroGrad();
	tests.expectTrue(accumulatingLayer.weight.grad.at({0, 0}) == 0.0 &&
		accumulatingLayer.bias.grad.at({0}) == 0.0,
		"zeroGrad clears the accumulated contributions");
	tests.expectTrue(
		accumulatingLayer.weight.grad.shape() == nnet::Tensor::Shape{1, 1} &&
		accumulatingLayer.bias.grad.shape() == nnet::Tensor::Shape{1},
		"clearing preserves both gradient shapes");

	const nnet::Tensor freshInputGradient =
		backwardThrough(accumulatingLayer, accumulationInput, accumulationUpstream);
	tests.expectNear(accumulatingLayer.weight.grad.at({0, 0}), 6.0, 1.0e-12,
		"backward after clearing starts a fresh weight gradient");
	tests.expectNear(accumulatingLayer.bias.grad.at({0}), 3.0, 1.0e-12,
		"backward after clearing starts a fresh bias gradient");
	tests.expectNear(freshInputGradient.at({0, 0}), 1.5, 1.0e-12,
		"backward after clearing still returns the same input gradient");
	tests.expectTrue(accumulatingLayer.weight.value.at({0, 0}) == 0.5 &&
		accumulatingLayer.bias.value.at({0}) == 0.25,
		"gradient clearing and backward leave weight and bias values unchanged");

    // Fixed shape means the construction shape, not merely value == grad.
    auto expectShapeRejected = [&](auto operation, const char* name) {
        bool rejected = false;
        try { operation(); }
        catch (const std::invalid_argument&) { rejected = true; }
        tests.expectTrue(rejected, name);
    };
    nnet::Dense fixed(1, 1);
    fixed.weight.value = nnet::Tensor({2, 1}, {0.5, 0.5});
    fixed.weight.grad = nnet::Tensor({2, 1}, {0.0, 0.0});
    expectShapeRejected([&] { (void)forwardOf(fixed, nnet::Tensor({1, 2}, {1.0, 2.0})); },
        "forward rejects jointly resized value and grad despite matching shapes");

    fixed.weight.value = nnet::Tensor({1, 1}, {0.5});
    fixed.weight.grad = nnet::Tensor({1, 1}, {4.0});
    fixed.bias.grad = nnet::Tensor({2}, {0.0, 0.0});
    fixed.bias.requiresGrad = false;
    // The check now happens as the forward pass builds each Parameter's leaf,
    // before any graph exists.
    expectShapeRejected([&] {
        (void)backwardThrough(fixed, nnet::Tensor({1, 1}, {2.0}),
                              nnet::Tensor({1, 1}, {3.0}));
    }, "a changed gradient shape is rejected even when frozen");
    tests.expectNear(fixed.weight.grad.at({0, 0}), 4.0, 0.0,
        "a rejected pass leaves earlier parameter gradients untouched");
    expectShapeRejected([&] { fixed.zeroGrad(); },
        "zeroGrad rejects an invalid parameter instead of silently repairing it");
    tests.expectNear(fixed.weight.grad.at({0, 0}), 4.0, 0.0,
        "zeroGrad validates all parameters before clearing any");

    fixed.bias.grad = nnet::Tensor({1}, {0.0});
    const nnet::Unary_Module& unary = fixed;
    tests.expectNear(unary.forward(nnet::makeLeaf(nnet::Tensor({1, 1}, {2.0})))->data.at({0, 0}),
        1.0, 1.0e-12, "public forward dispatches through the protected implementation hook");

    // ---- Dense under autograd ------------------------------------------------
    nnet::Dense observed(1, 1);
    observed.weight.value = nnet::Tensor({1, 1}, {0.5});
    observed.bias.value = nnet::Tensor({1}, {0.25});
    const nnet::Tensor one({1, 1}, {2.0});

    // A forward pass under NoGrad builds nothing, so backward from its output
    // cannot reach the Parameters at all.
    observed.zeroGrad();
    nnet::Value untracked;
    {
        nnet::NoGrad noGrad;
        untracked = observed.forward(nnet::makeLeaf(one));
    }
    nnet::backward(untracked);
    tests.expectTrue(observed.weight.grad.at({0, 0}) == 0.0 &&
        observed.bias.grad.at({0}) == 0.0,
        "a forward pass under NoGrad never reaches the parameter gradients");
    tests.expectNear(untracked->data.at({0, 0}), 1.25, 1.0e-12,
        "a forward pass under NoGrad still computes the same output");

    // One graph, one backward. A second call must not deposit again.
    observed.zeroGrad();
    const nnet::Value tracked = observed.forward(nnet::makeLeaf(one));
    nnet::backward(tracked);
    bool rejected = false;
    try { nnet::backward(tracked); }
    catch (const std::logic_error&) { rejected = true; }
    tests.expectTrue(rejected, "a second backward through a Dense graph is refused");
    tests.expectNear(observed.weight.grad.at({0, 0}), 2.0, 1.0e-12,
        "the refused backward leaves the weight gradient at a single contribution");
}

namespace {
class DiscoveryFixture : public nnet::Module {
public:
    std::vector<nnet::NamedParameter> local;
    std::vector<nnet::NamedModule> children;

protected:
    std::vector<nnet::NamedParameter> localNamedParameters() override {
        return local;
    }
    std::vector<nnet::NamedModule> namedChildren() override {
        return children;
    }
};
}

void runNamedParameterTests(TestRunner& tests) {
    tests.section("NAMED PARAMETER DISCOVERY");

    auto first = std::make_unique<nnet::Dense>(2, 2);
    auto last = std::make_unique<nnet::Dense>(2, 1);
    const std::vector<nnet::Parameter*> expectedPointers = {
        &first->weight, &first->bias, &last->weight, &last->bias
    };
    const auto local = first->namedParameters();
    tests.expectTrue(local.size() == 2 &&
        local[0].name == "weight" && local[0].parameter == &first->weight &&
        local[1].name == "bias" && local[1].parameter == &first->bias,
        "Dense names its actual weight and bias");

    nnet::Sigmoid activation;
    tests.expectTrue(activation.namedParameters().empty() &&
        activation.parameters().empty(),
        "Sigmoid discovers no parameters");

    std::vector<std::unique_ptr<nnet::Unary_Module>> components;
    components.push_back(std::move(first));
    components.push_back(std::make_unique<nnet::Sigmoid>());
    components.push_back(std::move(last));
    components.push_back(std::make_unique<nnet::Sigmoid>());
    auto inner = std::make_unique<nnet::Sequential>(std::move(components));

    const auto flat = inner->namedParameters();
    std::vector<std::string> names;
    std::vector<nnet::Parameter*> pointers;
    for (const auto& entry : flat) {
        names.push_back(entry.name);
        pointers.push_back(entry.parameter);
    }
    tests.expectTrue(names == std::vector<std::string>{
        "0.weight", "0.bias", "2.weight", "2.bias"},
        "Sequential prefixes names with component indices including activations");
    tests.expectTrue(pointers == expectedPointers,
        "named discovery preserves the original Parameter identities");

    const auto again = inner->namedParameters();
    bool repeatable = again.size() == flat.size();
    for (std::size_t i = 0; repeatable && i < flat.size(); ++i) {
        repeatable = again[i].name == flat[i].name &&
            again[i].parameter == flat[i].parameter;
    }
    tests.expectTrue(repeatable, "repeated discovery returns stable names and order");

    std::vector<std::unique_ptr<nnet::Unary_Module>> outerComponents;
    outerComponents.push_back(std::move(inner));
    nnet::Sequential outer(std::move(outerComponents));
    nnet::Module& base = outer;
    names.clear();
    for (const auto& entry : base.namedParameters()) {
        names.push_back(entry.name);
    }
    tests.expectTrue(names == std::vector<std::string>{
        "0.0.weight", "0.0.bias", "0.2.weight", "0.2.bias"},
        "nested Sequential recursively builds complete names through Module");
    tests.expectTrue(base.parameters() == expectedPointers,
        "pointer-only discovery uses the same recursive order");

    std::vector<nnet::Tensor> originalValues;
    for (nnet::Parameter* parameter : expectedPointers) {
        originalValues.push_back(parameter->value);
        parameter->grad = nnet::fill(parameter->grad.shape(), 3.0);
    }
    base.zeroGrad();
    bool cleared = true;
    bool preserved = true;
    for (std::size_t i = 0; i < expectedPointers.size(); ++i) {
        const auto* parameter = expectedPointers[i];
        cleared = cleared && parameter->grad.shape() == parameter->value.shape() &&
            parameter->grad.getData() == std::vector<double>(parameter->grad.numel(), 0.0);
        preserved = preserved && parameter->value.shape() == originalValues[i].shape() &&
            parameter->value.getData() == originalValues[i].getData();
    }
    tests.expectTrue(cleared, "zeroGrad clears every nested parameter through named discovery");
    tests.expectTrue(preserved, "recursive clearing preserves all parameter values");

    const auto expectRejected = [&tests](nnet::Module& module, const char* message) {
        bool rejected = false;
        try {
            (void)module.namedParameters();
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        tests.expectTrue(rejected, message);
    };

    DiscoveryFixture invalid;
    invalid.local = {{"", expectedPointers[0]}};
    expectRejected(invalid, "discovery rejects empty local names");
    invalid.local = {{"a.b", expectedPointers[0]}};
    expectRejected(invalid, "discovery rejects dots in local names");
    invalid.local = {{"weight", expectedPointers[0]}, {"weight", expectedPointers[1]}};
    expectRejected(invalid, "discovery rejects duplicate local parameter names");
    invalid.local = {{"weight", nullptr}};
    expectRejected(invalid, "discovery rejects null Parameter pointers");
    invalid.local = {{"first", expectedPointers[0]}, {"second", expectedPointers[0]}};
    expectRejected(invalid, "discovery rejects duplicate Parameter identities");

    invalid.local.clear();
    DiscoveryFixture childA;
    DiscoveryFixture childB;
    invalid.children = {{"child", &childA}, {"child", &childB}};
    expectRejected(invalid, "discovery rejects duplicate child names");
    invalid.children = {{"child", nullptr}};
    expectRejected(invalid, "discovery rejects null child pointers");
    invalid.children = {{"self", &invalid}};
    expectRejected(invalid, "discovery rejects a cycle without recursive overflow");
    invalid.children = {{"first", &childA}, {"second", &childA}};
    expectRejected(invalid, "discovery rejects a module registered under two parents or names");
}
