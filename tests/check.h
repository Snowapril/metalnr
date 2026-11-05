// Minimal test helpers: CHECK, CHECK_NEAR, checkSummary. No framework.
#pragma once

#include <cmath>
#include <cstdio>

inline int gCheckCount = 0;
inline int gCheckFailures = 0;

inline void checkRecord(bool ok, const char* expr, const char* file, int line) {
    ++gCheckCount;
    if (ok) return;
    ++gCheckFailures;
    std::fprintf(stderr, "%s:%d: CHECK failed: %s\n", file, line, expr);
}

inline void checkNearRecord(double a, double b, double tol, const char* expr, const char* file, int line) {
    ++gCheckCount;
    if (std::fabs(a - b) <= tol) return;  // NaN fails
    ++gCheckFailures;
    std::fprintf(stderr, "%s:%d: CHECK_NEAR failed: %s (%.9g vs %.9g, tol %.3g)\n", file, line, expr, a, b, tol);
}

#define CHECK(cond) checkRecord(static_cast<bool>(cond), #cond, __FILE__, __LINE__)
#define CHECK_NEAR(a, b, tol) \
    checkNearRecord(static_cast<double>(a), static_cast<double>(b), static_cast<double>(tol), #a " ~ " #b, __FILE__, __LINE__)

// Prints the totals; returns the process exit code (0 when every check passed).
inline int checkSummary() {
    std::fprintf(stderr, "%d checks, %d failed\n", gCheckCount, gCheckFailures);
    return gCheckFailures == 0 ? 0 : 1;
}
