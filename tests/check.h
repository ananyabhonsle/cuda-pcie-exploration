#pragma once
#include <stdio.h>

// Minimal test harness: CHECK records failures, check_finish() reports them.
static int g_check_failures = 0;

#define CHECK(cond)                                                        \
    do {                                                                   \
        if (!(cond)) {                                                     \
            printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);         \
            g_check_failures++;                                            \
        }                                                                  \
    } while (0)

// Prints the summary and returns the process exit code.
inline int check_finish() {
    if (g_check_failures == 0) {
        printf("ALL PASSED\n");
        return 0;
    }
    printf("%d check(s) FAILED\n", g_check_failures);
    return 1;
}
