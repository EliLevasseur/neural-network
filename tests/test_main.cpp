// Codex wrote this automated test entry point.

#include "test_utils.h"

void runNetworkTests(TestRunner& tests);
void runTrainingTests(TestRunner& tests);
void runTensorTests(TestRunner& tests);
void runDataFrameTests(TestRunner& tests);
void runOperationTests(TestRunner& tests);

int main() {
    TestRunner tests;

    runNetworkTests(tests);
    runTrainingTests(tests);
    runTensorTests(tests);
    runDataFrameTests(tests);
    runOperationTests(tests);

    return tests.finish();
}
