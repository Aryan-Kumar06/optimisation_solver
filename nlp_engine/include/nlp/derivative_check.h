#pragma once
#include "nlp/model.h"
namespace nlp {
struct DerivativeCheck {
    bool passed = false;
    double maximumRelativeError = 0;
    int row = -1; // -1: objective; otherwise constraint index
    int column = -1;
    std::string message;
};
// Central differences in original units. This diagnostic deliberately does not
// project perturbations: at a domain/boundary failure it reports unavailable,
// rather than claiming the derivative is wrong. Cost: 2*n+1 evaluations.
DerivativeCheck checkDerivatives(const Problem& problem,
    const std::vector<double>& x, double relativeTolerance = 1e-5,
    double relativeStep = 1e-5);
}
