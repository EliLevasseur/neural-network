#include "test_utils.h"

void runNetworkTests(TestRunner& tests);
void runTrainingTests(TestRunner& tests);

int main() {
    TestRunner tests;

    runNetworkTests(tests);
    runTrainingTests(tests);

    return tests.finish();
}
