#include "nlp/derivative_check.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>
namespace nlp {
DerivativeCheck checkDerivatives(const Problem& problem,
    const std::vector<double>& x, double tolerance, double step) {
    DerivativeCheck result;
    try {
        if (!std::isfinite(tolerance) || tolerance <= 0 || !std::isfinite(step) || step <= 0)
            throw std::invalid_argument("invalid derivative check options");
        const size_t n = problem.variableBounds().size();
        const size_t m = problem.constraintBounds().size();
        if (x.size() != n) throw std::invalid_argument("point dimension mismatch");
        auto validate = [&](const Evaluation& e) {
            if (e.gradient.size() != n || e.constraints.size() != m ||
                e.jacobian.rows() != static_cast<int>(m) || e.jacobian.columns() != static_cast<int>(n) || !e.jacobian.validate())
                throw std::invalid_argument("derivative dimensions or structure invalid");
            if (!std::isfinite(e.objective)) throw std::domain_error("non-finite objective");
            for (const auto* v : {&e.gradient, &e.constraints, &e.jacobian.csrValues()})
                for (double a : *v) if (!std::isfinite(a)) throw std::domain_error("non-finite derivative or constraint");
        };
        auto base = problem.evaluate(x); validate(base);
        for (size_t j = 0; j < n; ++j) {
            if (!std::isfinite(x[j])) throw std::invalid_argument("point must be finite");
            double h = step * std::max(1.0, std::abs(x[j]));
            auto plus = x, minus = x;
            plus[j] += h; minus[j] -= h;
            if (!std::isfinite(plus[j]) || !std::isfinite(minus[j]) || plus[j] == minus[j])
                throw std::domain_error("invalid finite difference perturbation");
            auto a = problem.evaluate(plus), b = problem.evaluate(minus);
            validate(a); validate(b);
            std::vector<double> unit(n, 0), column;
            unit[j] = 1; base.jacobian.multiply(unit, column);
            auto compare = [&](double analytic, double numerical, int row) {
                if (!std::isfinite(numerical)) throw std::domain_error("non-finite difference quotient");
                double error = std::abs(analytic - numerical) / std::max({1.0, std::abs(analytic), std::abs(numerical)});
                if (error > result.maximumRelativeError) {
                    result.maximumRelativeError = error;
                    result.row = row; result.column = static_cast<int>(j);
                }
            };
            double denominator = plus[j] - minus[j];
            compare(base.gradient[j], (a.objective - b.objective) / denominator, -1);
            for (size_t i = 0; i < m; ++i)
                compare(column[i], (a.constraints[i] - b.constraints[i]) / denominator, static_cast<int>(i));
        }
        result.passed = result.maximumRelativeError <= tolerance;
        result.message = result.passed ? "derivatives agree with central differences" : "derivative mismatch";
    } catch (const std::exception& e) {
        result.passed = false; result.message = e.what();
    }
    return result;
}
}
