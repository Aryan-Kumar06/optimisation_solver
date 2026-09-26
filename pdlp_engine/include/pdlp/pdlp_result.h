#pragma once

#include "pdlp/compute_backend.h"

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace pdlp {

enum class PdlpStatus {
    Optimal,

    // Proven primal infeasible, with a Farkas ray in `dualRay`.
    Infeasible,

    // Proven dual infeasible: the primal is unbounded along `primalRay`,
    // provided it is feasible.
    Unbounded,

    IterationLimit,
    TimeLimit,
    NumericalFailure,
    InvalidProblem
};

struct PdlpResult {
    PdlpStatus status = PdlpStatus::InvalidProblem;
    std::string statusMessage;

    std::vector<double> primal;
    std::vector<double> rowDual;

    // Populated for Infeasible and Unbounded respectively, in the original
    // problem's coordinates. A branch-and-bound layer needs only the status to
    // prune a node; the rays are what Benders cuts and user-facing proofs need.
    std::vector<double> dualRay;
    std::vector<double> primalRay;

    double primalObjective = std::numeric_limits<double>::quiet_NaN();
    double dualObjective = -std::numeric_limits<double>::infinity();

    double primalResidual = std::numeric_limits<double>::infinity();
    double dualResidual = std::numeric_limits<double>::infinity();
    double relativeGap = std::numeric_limits<double>::infinity();

    std::int64_t iterations = 0;

    // Total trial steps computed, including those the linesearch rejected. Each
    // trial costs one pass over the matrix, so this -- not `iterations` -- is the
    // honest measure of work done.
    std::int64_t stepTrials = 0;

    int restartCount = 0;
    double solveTimeSeconds = 0.0;

    // Diagnostics. finalStepSize / staticStepBound above 1 means the adaptive
    // linesearch found the local curvature to be milder than the worst-case
    // Pock-Chambolle bound, which is the whole point of running it.
    double finalStepSize = 0.0;
    double staticStepBound = 0.0;
    double finalPrimalWeight = 1.0;

    // Which backend ran the iteration, and why that one. A solve that asked
    // for Auto reports what Auto resolved to; the message names the reason
    // (threshold, availability, fallback) so "did the GPU run?" is never a
    // guess.
    ComputeBackend executedBackend = ComputeBackend::Cpu;
    std::string backendMessage;

    // Device setup and transfer accounting (zero on the CPU), plus the time
    // spent in host-side termination and certificate checks on any backend.
    BackendProfile backendProfile;
    double hostCheckSeconds = 0.0;
};

[[nodiscard]] const char* toString(PdlpStatus status) noexcept;

}  // namespace pdlp

