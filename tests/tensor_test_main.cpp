#include "test_utils.h"

void runTensorTests(TestRunner& tests);
void runDataFrameTests(TestRunner& tests);
void runOperationTests(TestRunner& tests);
void runDenseTests(TestRunner& tests);
void runNamedParameterTests(TestRunner& tests);
void runAutogradTests(TestRunner& tests);

int main() {
    TestRunner tests;

    runTensorTests(tests);
    runDataFrameTests(tests);
    runOperationTests(tests);
    runDenseTests(tests);
    runNamedParameterTests(tests);
    runAutogradTests(tests);

    return tests.finish();
}
