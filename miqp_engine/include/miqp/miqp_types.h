#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace miqp {

enum class MiqpStatus {
    Optimal,
    Infeasible,
    NodeLimit,
    TimeLimit,
    RelaxationFailure
};

struct MiqpOptions {
    double timeLimitSeconds = 0.0;
    std::int64_t nodeLimit = 0;
    double integralityTolerance = 1e-6;
    double feasibilityTolerance = 1e-7;
    double objectiveTolerance = 1e-8;
    int threadCount = 0;
};

struct MiqpResult {
    MiqpStatus status = MiqpStatus::RelaxationFailure;
    std::string message;
    double objectiveValue = std::numeric_limits<double>::quiet_NaN();
    std::vector<double> primal;
    std::int64_t nodeCount = 0;
    std::int64_t qpIterations = 0;
};

}  // namespace miqp
