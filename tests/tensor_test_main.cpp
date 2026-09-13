#include "test_utils.h"

void runTensorTests(TestRunner& tests);
void runDataFrameTests(TestRunner& tests);
void runOperationTests(TestRunner& tests);

int main() {
    TestRunner tests;

    runTensorTests(tests);
    runDataFrameTests(tests);
    runOperationTests(tests);

    return tests.finish();
}
