#include "test_utils.h"

void runParityTests(TestRunner& tests);

int main() {
    TestRunner tests;

    runParityTests(tests);

    return tests.finish();
}
