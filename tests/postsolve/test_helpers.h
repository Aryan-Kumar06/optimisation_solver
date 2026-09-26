#pragma once

#include "model/model.h"
#include "presolve/presolve_result.h"
#include "postsolve/postsolver.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace test_helpers {

// Reuses the project's existing postsolve tolerance rather than
// introducing a second, competing constant (see postsolve/postsolver.h).
constexpr double kTolerance = postsolve::DEFAULT_POSTSOLVE_TOLERANCE;

inline bool nearlyEqual(double a, double b, double tolerance = kTolerance) {
    return std::fabs(a - b) <= tolerance;
}

// ============================================================
// Model-building helpers
// ============================================================

inline model::Variable makeVariable(
    const std::string& name,
    double lowerBound,
    double upperBound,
    model::VariableType type = model::VariableType::Continuous
) {
    model::Variable v;
    v.name = name;
    v.lowerBound = lowerBound;
    v.upperBound = upperBound;
    v.type = type;
    return v;
}

inline model::Constraint makeConstraint(
    const std::string& name,
    double lowerBound,
    double upperBound,
    std::vector<model::LinearTerm> terms
) {
    model::Constraint c;
    c.name = name;
    c.lowerBound = lowerBound;
    c.upperBound = upperBound;
    c.linearTerms = std::move(terms);
    return c;
}

inline model::LinearTerm term(int variableIndex, double value) {
    return model::LinearTerm{variableIndex, value};
}

// ============================================================
// Evaluation helpers
// ============================================================

inline double evaluateConstraintActivity(
    const model::Constraint& constraint,
    const std::vector<double>& solution
) {
    double activity = 0.0;

    for (const auto& t : constraint.linearTerms) {
        activity += t.value * solution[static_cast<std::size_t>(t.variableIndex)];
    }

    return activity;
}

inline bool constraintSatisfied(
    const model::Constraint& constraint,
    const std::vector<double>& solution,
    double tolerance = kTolerance
) {
    double activity = evaluateConstraintActivity(constraint, solution);

    return (activity >= constraint.lowerBound - tolerance) &&
           (activity <= constraint.upperBound + tolerance);
}

inline bool boundSatisfied(
    const model::Variable& variable,
    double value,
    double tolerance = kTolerance
) {
    return (value >= variable.lowerBound - tolerance) &&
           (value <= variable.upperBound + tolerance);
}

// ============================================================
// Test reporting helpers (useful failure messages)
// ============================================================

inline void reportFailure(
    const std::string& testName,
    const std::string& expected,
    const std::string& actual
) {
    std::cerr << "[FAIL] " << testName << "\n"
              << "  Expected: " << expected << "\n"
              << "  Actual:   " << actual << "\n";
}

inline void checkNearlyEqual(
    const std::string& testName,
    const std::string& label,
    double expected,
    double actual,
    double tolerance = kTolerance
) {
    if (!nearlyEqual(expected, actual, tolerance)) {
        std::cerr << "[FAIL] " << testName << " — " << label << "\n"
                  << "  Expected " << label << " = " << expected << "\n"
                  << "  Got      " << label << " = " << actual << "\n";
        std::abort();
    }
}

inline void checkTrue(
    const std::string& testName,
    const std::string& label,
    bool condition
) {
    if (!condition) {
        std::cerr << "[FAIL] " << testName << " — " << label << " was false\n";
        std::abort();
    }
}

} // namespace test_helpers