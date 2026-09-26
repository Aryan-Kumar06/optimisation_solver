#include "model/model.h"
#include "presolve/presolver.h"
#include "postsolve/postsolver.h"
#include "test_helpers.h"

#include <iostream>
#include <limits>
#include <string>
#include <vector>

using namespace test_helpers;

// ============================================================
// PART 1: Presolve Metadata Tests (PR #4 baseline)
// ============================================================

// ------------------------------------------------------------
// Fixed variable recorded in PostsolveMetadata
// ------------------------------------------------------------
void test_fixed_variable_recorded() {
    const std::string testName = "test_fixed_variable_recorded";

    model::Model original;
    original.name = "FixedVariableModel";

    original.variables.push_back(makeVariable("x", 5.0, 5.0));
    original.variables.push_back(makeVariable("y", 0.0, 10.0));

    original.constraints.push_back(
        makeConstraint("C1", -100.0, 100.0, { term(0, 1.0), term(1, 1.0) })
    );

    presolve::Presolver presolver;
    presolve::PresolveResult result = presolver.run(original);

    checkTrue(testName, "model is not infeasible", !result.infeasible);

    checkTrue(
        testName,
        "at least one fixed variable was recorded",
        !result.postsolve.fixedVariables.empty()
    );

    bool foundX = false;
    for (const auto& fv : result.postsolve.fixedVariables) {
        if (fv.originalIndex == 0) {
            foundX = true;
            checkNearlyEqual(testName, "fixedValue for x", 5.0, fv.fixedValue);
            checkTrue(testName, "recorded name is 'x'", fv.name == "x");
        }
    }

    checkTrue(testName, "fixed variable record for original index 0 (x) found",
              foundX);

    checkTrue(
        testName,
        "originalToPresolvedVar has an entry for x",
        result.postsolve.originalToPresolvedVar.size() > 0
    );

    if (!result.postsolve.originalToPresolvedVar.empty()) {
        checkTrue(
            testName,
            "x (original index 0) maps to -1 (eliminated) in originalToPresolvedVar",
            result.postsolve.originalToPresolvedVar[0] == -1
        );
    }

    std::cout << "[PASS] " << testName << "\n";
}

// ------------------------------------------------------------
// Singleton equality — positive and negative coefficient
// ------------------------------------------------------------
void test_singleton_equality_positive_coefficient() {
    const std::string testName = "test_singleton_equality_positive_coefficient";

    model::Model original;
    original.name = "SingletonPositive";
    original.variables.push_back(makeVariable("x", -1000.0, 1000.0));
    original.constraints.push_back(
        makeConstraint("C1", 10.0, 10.0, { term(0, 2.0) })  // 2x = 10
    );

    presolve::Presolver presolver;
    presolve::PresolveResult result = presolver.run(original);

    checkTrue(testName, "model is not infeasible", !result.infeasible);

    bool found = false;
    for (const auto& fv : result.postsolve.fixedVariables) {
        if (fv.originalIndex == 0) {
            found = true;
            checkNearlyEqual(testName, "x from 2x=10", 5.0, fv.fixedValue);
        }
    }
    checkTrue(testName, "x was fixed via singleton row", found);

    std::cout << "[PASS] " << testName << "\n";
}

void test_singleton_equality_negative_coefficient() {
    const std::string testName = "test_singleton_equality_negative_coefficient";

    model::Model original;
    original.name = "SingletonNegative";
    original.variables.push_back(makeVariable("x", -1000.0, 1000.0));
    original.constraints.push_back(
        makeConstraint("C1", -10.0, -10.0, { term(0, -2.0) })  // -2x = -10
    );

    presolve::Presolver presolver;
    presolve::PresolveResult result = presolver.run(original);

    checkTrue(testName, "model is not infeasible", !result.infeasible);

    bool found = false;
    for (const auto& fv : result.postsolve.fixedVariables) {
        if (fv.originalIndex == 0) {
            found = true;
            checkNearlyEqual(
                testName,
                "x from -2x=-10 (sign check)",
                5.0,
                fv.fixedValue
            );
        }
    }
    checkTrue(testName, "x was fixed via singleton row with negative coefficient",
              found);

    std::cout << "[PASS] " << testName << "\n";
}

// ------------------------------------------------------------
// Bound tightening recorded
// ------------------------------------------------------------
void test_bound_tightening_recorded() {
    const std::string testName = "test_bound_tightening_recorded";

    model::Model original;
    original.name = "BoundTighteningModel";
    original.variables.push_back(makeVariable("x", 0.0, 20.0));

    original.constraints.push_back(
        makeConstraint("C1", 5.0, 20.0, { term(0, 1.0) })
    );

    presolve::Presolver presolver;
    presolve::PresolveResult result = presolver.run(original);

    checkTrue(testName, "model is not infeasible", !result.infeasible);

    checkNearlyEqual(
        testName,
        "original.variables[0].lowerBound unchanged",
        0.0,
        original.variables[0].lowerBound
    );
    checkNearlyEqual(
        testName,
        "original.variables[0].upperBound unchanged",
        20.0,
        original.variables[0].upperBound
    );

    bool foundTightening = false;
    for (const auto& t : result.transformations) {
        if (
            t.type == presolve::TransformationType::TightenLowerBound &&
            t.originalVariableIndex == 0
        ) {
            foundTightening = true;
            checkNearlyEqual(
                testName,
                "tightened lower bound new value",
                5.0,
                t.newValue
            );
        }
    }

    checkTrue(
        testName,
        "a TightenLowerBound transformation for x was recorded",
        foundTightening
    );

    std::cout << "[PASS] " << testName << "\n";
}

// ------------------------------------------------------------
// No-transformation baseline
// ------------------------------------------------------------
void test_no_transformations_baseline() {
    const std::string testName = "test_no_transformations_baseline";

    model::Model original;
    original.name = "NoOpModel";
    original.variables.push_back(makeVariable("x", 1.0, 2.0));
    original.variables.push_back(makeVariable("y", 1.0, 2.0));

    original.constraints.push_back(
        makeConstraint("C1", -100.0, 6.0, { term(0, 1.0), term(1, 3.0) })
    );

    presolve::Presolver presolver;
    presolve::PresolveResult result = presolver.run(original);

    checkTrue(testName, "model is not infeasible", !result.infeasible);

    checkTrue(
        testName,
        "presolvedVariables equals originalVariables (no elimination expected)",
        result.presolvedVariables == result.originalVariables
    );

    checkTrue(
        testName,
        "presolvedConstraints equals originalConstraints (no elimination expected)",
        result.presolvedConstraints == result.originalConstraints
    );

    std::cout << "[PASS] " << testName << "\n";
}

// ============================================================
// PART 2: Real Postsolve Regression Tests (Calling Postsolver::process)
// ============================================================

// ------------------------------------------------------------
// A. Fixed variable restoration
// ------------------------------------------------------------
void test_fixed_variable_postsolve() {
    const std::string testName = "test_fixed_variable_postsolve";

    model::Model original;
    original.name = "FixedVariablePostsolve";

    // x is fixed by bounds [5.0, 5.0], y is free in [0.0, 10.0]
    original.variables.push_back(makeVariable("x", 5.0, 5.0));
    original.variables.push_back(makeVariable("y", 0.0, 10.0));

    original.objective.linearTerms.push_back(term(0, 2.0));
    original.objective.linearTerms.push_back(term(1, 3.0));

    // Constraint: x + y <= 12.0
    original.constraints.push_back(
        makeConstraint("C1", -1e9, 12.0, { term(0, 1.0), term(1, 1.0) })
    );

    presolve::Presolver presolver;
    presolve::PresolveResult presolveRes = presolver.run(original);

    checkTrue(testName, "presolve succeeded", !presolveRes.infeasible);
    checkTrue(testName, "reduced model has 1 variable", presolveRes.model.variables.size() == 1);

    // Suppose the reduced solver returned y = 4.0
    std::vector<double> reducedSolution = {4.0};

    postsolve::Postsolver postsolver;
    postsolve::PostsolveResult result = postsolver.process(original, presolveRes, reducedSolution);

    checkTrue(testName, "postsolve reported success", result.isSuccess());
    checkTrue(testName, "returned vector has original-model size",
              result.primalSolution.size() == original.variables.size());

    // The fixed variable is restored to the correct original value
    checkNearlyEqual(testName, "restored x equals 5.0", 5.0, result.primalSolution[0]);
    checkNearlyEqual(testName, "restored y equals 4.0", 4.0, result.primalSolution[1]);

    // The reconstructed solution satisfies the original model
    checkTrue(testName, "x satisfies bounds", boundSatisfied(original.variables[0], result.primalSolution[0]));
    checkTrue(testName, "y satisfies bounds", boundSatisfied(original.variables[1], result.primalSolution[1]));
    checkTrue(testName, "C1 satisfied", constraintSatisfied(original.constraints[0], result.primalSolution));

    // Objective: 2*5.0 + 3*4.0 = 22.0
    checkNearlyEqual(testName, "objective value", 22.0, result.originalObjectiveValue);

    std::cout << "[PASS] " << testName << "\n";
}

// ------------------------------------------------------------
// B. Singleton equality restoration
// ------------------------------------------------------------
void test_singleton_equality_postsolve() {
    const std::string testName = "test_singleton_equality_postsolve";

    model::Model original;
    original.name = "SingletonEqualityPostsolve";

    original.variables.push_back(makeVariable("x", -1000.0, 1000.0));
    original.variables.push_back(makeVariable("y", 0.0, 20.0));

    original.objective.linearTerms.push_back(term(0, 10.0));
    original.objective.linearTerms.push_back(term(1, 1.0));

    // 2x = 10 -> fixes x = 5
    original.constraints.push_back(
        makeConstraint("C_singleton", 10.0, 10.0, { term(0, 2.0) })
    );
    // x + y <= 30
    original.constraints.push_back(
        makeConstraint("C_coupling", -1e9, 30.0, { term(0, 1.0), term(1, 1.0) })
    );

    presolve::Presolver presolver;
    presolve::PresolveResult presolveRes = presolver.run(original);

    checkTrue(testName, "presolve succeeded", !presolveRes.infeasible);
    checkTrue(testName, "x was removed from reduced model", presolveRes.model.variables.size() == 1);

    // Suppose solver found y = 7.0 for the remaining variable
    std::vector<double> reducedSolution = {7.0};

    postsolve::Postsolver postsolver;
    postsolve::PostsolveResult result = postsolver.process(original, presolveRes, reducedSolution);

    checkTrue(testName, "postsolve succeeded", result.isSuccess());
    checkTrue(testName, "primalSolution size equals 2", result.primalSolution.size() == 2);
    checkNearlyEqual(testName, "x restored to 5.0", 5.0, result.primalSolution[0]);
    checkNearlyEqual(testName, "y restored to 7.0", 7.0, result.primalSolution[1]);

    checkTrue(testName, "singleton constraint satisfied",
              constraintSatisfied(original.constraints[0], result.primalSolution));
    checkTrue(testName, "coupling constraint satisfied",
              constraintSatisfied(original.constraints[1], result.primalSolution));

    checkNearlyEqual(testName, "objective value", 57.0, result.originalObjectiveValue);

    std::cout << "[PASS] " << testName << "\n";
}

// ------------------------------------------------------------
// C. Negative singleton coefficient
// ------------------------------------------------------------
void test_negative_singleton_postsolve() {
    const std::string testName = "test_negative_singleton_postsolve";

    model::Model original;
    original.name = "NegativeSingletonPostsolve";

    original.variables.push_back(makeVariable("x", -1000.0, 1000.0));
    original.variables.push_back(makeVariable("y", 0.0, 10.0));

    original.objective.linearTerms.push_back(term(0, 1.0));
    original.objective.linearTerms.push_back(term(1, 2.0));

    // -2x = -10 -> x = 5 (protects against sign errors)
    original.constraints.push_back(
        makeConstraint("C_neg_singleton", -10.0, -10.0, { term(0, -2.0) })
    );
    // x - y >= 0
    original.constraints.push_back(
        makeConstraint("C_bound", 0.0, 1e9, { term(0, 1.0), term(1, -1.0) })
    );

    presolve::Presolver presolver;
    presolve::PresolveResult presolveRes = presolver.run(original);

    checkTrue(testName, "presolve succeeded", !presolveRes.infeasible);

    // Reduced model has y
    std::vector<double> reducedSolution = {3.0};

    postsolve::Postsolver postsolver;
    postsolve::PostsolveResult result = postsolver.process(original, presolveRes, reducedSolution);

    checkTrue(testName, "postsolve succeeded", result.isSuccess());
    checkNearlyEqual(testName, "x restored to positive 5.0 (sign check)", 5.0, result.primalSolution[0]);
    checkNearlyEqual(testName, "y restored to 3.0", 3.0, result.primalSolution[1]);

    checkTrue(testName, "singleton row satisfied",
              constraintSatisfied(original.constraints[0], result.primalSolution));
    checkTrue(testName, "bound constraint satisfied",
              constraintSatisfied(original.constraints[1], result.primalSolution));

    std::cout << "[PASS] " << testName << "\n";
}

// ------------------------------------------------------------
// D. Bound tightening
// ------------------------------------------------------------
void test_bound_tightening_postsolve() {
    const std::string testName = "test_bound_tightening_postsolve";

    model::Model original;
    original.name = "BoundTighteningPostsolve";

    original.variables.push_back(makeVariable("x", 0.0, 20.0));
    original.variables.push_back(makeVariable("y", 0.0, 20.0));

    original.objective.linearTerms.push_back(term(0, 2.0));
    original.objective.linearTerms.push_back(term(1, 1.0));

    // 1*x in [5.0, 20.0] -> tightens lower bound of x from 0 to 5
    original.constraints.push_back(
        makeConstraint("C_tighten", 5.0, 20.0, { term(0, 1.0) })
    );
    original.constraints.push_back(
        makeConstraint("C_sum", 0.0, 30.0, { term(0, 1.0), term(1, 1.0) })
    );

    presolve::Presolver presolver;
    presolve::PresolveResult presolveRes = presolver.run(original);

    checkTrue(testName, "presolve succeeded", !presolveRes.infeasible);
    checkTrue(testName, "both variables survive in reduced model",
              presolveRes.model.variables.size() == 2);

    // Reduced solution respecting the tightened bound (x = 6.0 >= 5.0, y = 4.0)
    std::vector<double> reducedSolution = {6.0, 4.0};

    postsolve::Postsolver postsolver;
    postsolve::PostsolveResult result = postsolver.process(original, presolveRes, reducedSolution);

    checkTrue(testName, "postsolve succeeded", result.isSuccess());
    checkTrue(testName, "primalSolution size equals 2", result.primalSolution.size() == 2);
    checkNearlyEqual(testName, "x reconstructed correctly", 6.0, result.primalSolution[0]);
    checkNearlyEqual(testName, "y reconstructed correctly", 4.0, result.primalSolution[1]);

    checkTrue(testName, "C_tighten satisfied", constraintSatisfied(original.constraints[0], result.primalSolution));
    checkTrue(testName, "C_sum satisfied", constraintSatisfied(original.constraints[1], result.primalSolution));
    checkTrue(testName, "x bound satisfied", boundSatisfied(original.variables[0], result.primalSolution[0]));
    checkTrue(testName, "y bound satisfied", boundSatisfied(original.variables[1], result.primalSolution[1]));

    std::cout << "[PASS] " << testName << "\n";
}

// ------------------------------------------------------------
// E. No-transformation baseline
// ------------------------------------------------------------
void test_no_transformation_postsolve() {
    const std::string testName = "test_no_transformation_postsolve";

    model::Model original;
    original.name = "NoTransformationPostsolve";

    original.variables.push_back(makeVariable("x", 1.0, 10.0));
    original.variables.push_back(makeVariable("y", 1.0, 10.0));

    original.objective.offset = 10.0;
    original.objective.linearTerms.push_back(term(0, 3.0));
    original.objective.linearTerms.push_back(term(1, 4.0));

    original.constraints.push_back(
        makeConstraint("C1", -100.0, 20.0, { term(0, 1.0), term(1, 3.0) })
    );

    presolve::Presolver presolver;
    presolve::PresolveResult presolveRes = presolver.run(original);

    checkTrue(testName, "presolve succeeded", !presolveRes.infeasible);
    checkTrue(testName, "no variables eliminated", presolveRes.presolvedVariables == 2);
    checkTrue(testName, "no constraints eliminated", presolveRes.presolvedConstraints == 1);

    std::vector<double> reducedSolution = {2.5, 3.5};

    postsolve::Postsolver postsolver;
    postsolve::PostsolveResult result = postsolver.process(original, presolveRes, reducedSolution);

    checkTrue(testName, "postsolve succeeded", result.isSuccess());
    checkTrue(testName, "original dimensions preserved", result.primalSolution.size() == 2);
    checkNearlyEqual(testName, "x equals reduced value", 2.5, result.primalSolution[0]);
    checkNearlyEqual(testName, "y equals reduced value", 3.5, result.primalSolution[1]);

    // 10.0 + 3*2.5 + 4*3.5 = 10.0 + 7.5 + 14.0 = 31.5
    checkNearlyEqual(testName, "objective value matches", 31.5, result.originalObjectiveValue);

    checkTrue(testName, "x bound satisfied", boundSatisfied(original.variables[0], result.primalSolution[0]));
    checkTrue(testName, "y bound satisfied", boundSatisfied(original.variables[1], result.primalSolution[1]));
    checkTrue(testName, "constraint satisfied", constraintSatisfied(original.constraints[0], result.primalSolution));

    std::cout << "[PASS] " << testName << "\n";
}

// ------------------------------------------------------------
// F. Invalid mapping
// ------------------------------------------------------------
void test_invalid_mapping_postsolve() {
    const std::string testName = "test_invalid_mapping_postsolve";

    model::Model original;
    original.name = "InvalidMappingModel";
    original.variables.push_back(makeVariable("x0", 0.0, 10.0));
    original.variables.push_back(makeVariable("x1", 0.0, 10.0));
    original.variables.push_back(makeVariable("x2", 0.0, 10.0));

    // Case 1: Incomplete variable coverage
    // Model has 3 variables, but mapping only covers 1 variable; x1 and x2 are neither mapped nor fixed.
    presolve::PresolveResult corruptedResult;
    corruptedResult.originalVariables = 3;
    corruptedResult.presolvedVariables = 1;
    corruptedResult.postsolve.presolvedToOriginalVar = {0};
    corruptedResult.postsolve.originalToPresolvedVar = {0, -1, -1};

    postsolve::Postsolver postsolver;
    postsolve::PostsolveResult result = postsolver.process(original, corruptedResult, {1.0});

    checkTrue(testName, "incomplete coverage fails postsolve", !result.isSuccess());
    checkTrue(testName, "status is InvalidMapping for incomplete coverage",
              result.status == postsolve::PostsolveStatus::InvalidMapping);
    checkTrue(testName, "does not produce a primal solution on invalid mapping",
              result.primalSolution.empty());

    // Case 2: Out of range original variable index
    presolve::PresolveResult outOfRangeResult;
    outOfRangeResult.originalVariables = 3;
    outOfRangeResult.presolvedVariables = 1;
    outOfRangeResult.postsolve.presolvedToOriginalVar = {99};
    outOfRangeResult.postsolve.originalToPresolvedVar = {-1, -1, -1};

    result = postsolver.process(original, outOfRangeResult, {1.0});
    checkTrue(testName, "out-of-range index fails postsolve", !result.isSuccess());
    checkTrue(testName, "status is InvalidMapping for out-of-range index",
              result.status == postsolve::PostsolveStatus::InvalidMapping);
    checkTrue(testName, "does not produce a primal solution on out-of-range mapping",
              result.primalSolution.empty());

    std::cout << "[PASS] " << testName << "\n";
}

// ------------------------------------------------------------
// G. Solution validation: non-finite primal values
// ------------------------------------------------------------
void test_solution_validation_non_finite_postsolve() {
    const std::string testName = "test_solution_validation_non_finite_postsolve";

    constexpr double INF = std::numeric_limits<double>::infinity();

    model::Model original;
    original.name = "NonFiniteValidationModel";
    // Free variable with [-INF, +INF] bounds: non-finite must still be rejected
    original.variables.push_back(makeVariable("x0", -INF, INF));

    presolve::PresolveResult identityResult;
    identityResult.originalVariables = 1;
    identityResult.presolvedVariables = 1;
    identityResult.postsolve.presolvedToOriginalVar = {0};
    identityResult.postsolve.originalToPresolvedVar = {0};

    postsolve::Postsolver postsolver;

    // 1. NaN
    double nanVal = std::numeric_limits<double>::quiet_NaN();
    postsolve::PostsolveResult resultNaN = postsolver.process(original, identityResult, {nanVal});
    checkTrue(testName, "NaN primal value rejected", !resultNaN.isSuccess());
    checkTrue(testName, "NaN reports BoundViolation",
              resultNaN.status == postsolve::PostsolveStatus::BoundViolation);
    checkTrue(testName, "NaN error message mentions non-finite",
              resultNaN.errorMessage.find("Non-finite") != std::string::npos);

    // 2. +Inf
    double posInf = INF;
    postsolve::PostsolveResult resultPosInf = postsolver.process(original, identityResult, {posInf});
    checkTrue(testName, "+Inf primal value rejected", !resultPosInf.isSuccess());
    checkTrue(testName, "+Inf reports BoundViolation",
              resultPosInf.status == postsolve::PostsolveStatus::BoundViolation);
    checkTrue(testName, "+Inf error message mentions non-finite",
              resultPosInf.errorMessage.find("Non-finite") != std::string::npos);

    // 3. -Inf
    double negInf = -INF;
    postsolve::PostsolveResult resultNegInf = postsolver.process(original, identityResult, {negInf});
    checkTrue(testName, "-Inf primal value rejected", !resultNegInf.isSuccess());
    checkTrue(testName, "-Inf reports BoundViolation",
              resultNegInf.status == postsolve::PostsolveStatus::BoundViolation);
    checkTrue(testName, "-Inf error message mentions non-finite",
              resultNegInf.errorMessage.find("Non-finite") != std::string::npos);

    std::cout << "[PASS] " << testName << "\n";
}

int main() {
    // Part 1: Metadata tests
    test_fixed_variable_recorded();
    test_singleton_equality_positive_coefficient();
    test_singleton_equality_negative_coefficient();
    test_bound_tightening_recorded();
    test_no_transformations_baseline();

    // Part 2: Real postsolve regression tests
    test_fixed_variable_postsolve();
    test_singleton_equality_postsolve();
    test_negative_singleton_postsolve();
    test_bound_tightening_postsolve();
    test_no_transformation_postsolve();
    test_invalid_mapping_postsolve();
    test_solution_validation_non_finite_postsolve();

    std::cout << "All postsolve metadata and regression tests passed successfully!" << std::endl;
    return 0;
}
