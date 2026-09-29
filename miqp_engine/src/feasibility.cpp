#include "miqp/feasibility.h"

#include <cmath>
#include <cstddef>
#include <sstream>

namespace miqp {
namespace {

bool below(double value, double lower, double tol) {
    return std::isfinite(lower) && value < lower - tol;
}

bool above(double value, double upper, double tol) {
    return std::isfinite(upper) && value > upper + tol;
}

}  // namespace

FeasibilityCheck checkFeasibility(const model::Model& model,
                                  const std::vector<double>& x,
                                  double feasibilityTolerance,
                                  double integralityTolerance) {
    FeasibilityCheck result;

    if (x.size() != model.variables.size()) {
        result.reason = "candidate dimension does not match model";
        return result;
    }

    for (std::size_t j = 0; j < x.size(); ++j) {
        const double value = x[j];
        const auto& variable = model.variables[j];

        if (!std::isfinite(value)) {
            result.reason = "candidate contains a non-finite variable value";
            return result;
        }

        if (below(value, variable.lowerBound, feasibilityTolerance) ||
            above(value, variable.upperBound, feasibilityTolerance)) {
            std::ostringstream out;
            out << "candidate violates bounds for variable " << j;
            result.reason = out.str();
            return result;
        }

        if (variable.type != model::VariableType::Continuous) {
            if (std::abs(value - std::round(value)) > integralityTolerance) {
                std::ostringstream out;
                out << "candidate violates integrality for variable " << j;
                result.reason = out.str();
                return result;
            }
        }

        if (variable.type == model::VariableType::Binary) {
            const bool nearZero =
                std::abs(value) <= integralityTolerance;
            const bool nearOne =
                std::abs(value - 1.0) <= integralityTolerance;

            if (!nearZero && !nearOne) {
                std::ostringstream out;
                out << "candidate violates binary domain for variable " << j;
                result.reason = out.str();
                return result;
            }
        }
    }

    for (std::size_t i = 0; i < model.constraints.size(); ++i) {
        const auto& constraint = model.constraints[i];

        double activity = 0.0;

        for (const auto& term : constraint.linearTerms) {
            const std::size_t j =
                static_cast<std::size_t>(term.variableIndex);

            if (j >= x.size()) {
                result.reason =
                    "constraint references an invalid variable index";
                return result;
            }

            activity += term.value * x[j];
        }

        if (below(activity,
                  constraint.lowerBound,
                  feasibilityTolerance) ||
            above(activity,
                  constraint.upperBound,
                  feasibilityTolerance)) {
            std::ostringstream out;
            out << "candidate violates constraint " << i;
            result.reason = out.str();
            return result;
        }
    }

    result.feasible = true;
    result.reason = "candidate is feasible";
    return result;
}

FeasibilityCheck checkRoundingSafety(
    const model::Model& model,
    const std::vector<double>& beforeRounding,
    const std::vector<double>& rounded,
    double crossingTolerance) {

    FeasibilityCheck result;

    if (beforeRounding.size() != model.variables.size() ||
        rounded.size() != model.variables.size()) {
        result.reason =
            "rounding-safety candidate dimension does not match model";
        return result;
    }

    for (std::size_t j = 0; j < rounded.size(); ++j) {
        const auto& variable = model.variables[j];

        const double before = beforeRounding[j];
        const double after = rounded[j];

        if (std::isfinite(variable.lowerBound) &&
            before >= variable.lowerBound - crossingTolerance &&
            after < variable.lowerBound - crossingTolerance) {
            std::ostringstream out;
            out << "rounding introduced a lower-bound violation for variable "
                << j;
            result.reason = out.str();
            return result;
        }

        if (std::isfinite(variable.upperBound) &&
            before <= variable.upperBound + crossingTolerance &&
            after > variable.upperBound + crossingTolerance) {
            std::ostringstream out;
            out << "rounding introduced an upper-bound violation for variable "
                << j;
            result.reason = out.str();
            return result;
        }
    }

    for (std::size_t i = 0; i < model.constraints.size(); ++i) {
        const auto& constraint = model.constraints[i];

        double beforeActivity = 0.0;
        double afterActivity = 0.0;

        for (const auto& term : constraint.linearTerms) {
            const std::size_t j =
                static_cast<std::size_t>(term.variableIndex);

            if (j >= rounded.size()) {
                result.reason =
                    "constraint references an invalid variable index";
                return result;
            }

            beforeActivity += term.value * beforeRounding[j];
            afterActivity += term.value * rounded[j];
        }

        if (std::isfinite(constraint.lowerBound) &&
            beforeActivity >= constraint.lowerBound - crossingTolerance &&
            afterActivity < constraint.lowerBound - crossingTolerance) {
            std::ostringstream out;
            out << "rounding introduced a lower-row violation for constraint "
                << i;
            result.reason = out.str();
            return result;
        }

        if (std::isfinite(constraint.upperBound) &&
            beforeActivity <= constraint.upperBound + crossingTolerance &&
            afterActivity > constraint.upperBound + crossingTolerance) {
            std::ostringstream out;
            out << "rounding introduced an upper-row violation for constraint "
                << i;
            result.reason = out.str();
            return result;
        }
    }

    result.feasible = true;
    result.reason =
        "rounding did not introduce a new bound or row violation";
    return result;
}

}  // namespace miqp