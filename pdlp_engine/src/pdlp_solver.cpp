#include "pdlp/pdlp_solver.h"

#include "pdlp/feasibility_polishing.h"
#include "pdlp/infeasibility.h"
#include "pdlp/iteration_backend.h"
#include "pdlp/parallel.h"
#include "pdlp/pdhg_kernel.h"
#include "pdlp/preconditioner.h"
#include "pdlp/restart_controller.h"
#include "pdlp/scaling.h"
#include "pdlp/step_controller.h"
#include "pdlp/termination.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <string>

namespace pdlp {
namespace {

using Clock = std::chrono::steady_clock;

double elapsedSeconds(const Clock::time_point& start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

double initialValue(double lower, double upper) {
    if (lower <= 0.0 && 0.0 <= upper) {
        return 0.0;
    }
    if (std::isfinite(lower)) {
        return lower;
    }
    if (std::isfinite(upper)) {
        return upper;
    }
    return 0.0;
}

PdlpResult makeResult(
    PdlpStatus status,
    std::string message,
    CandidateIterate candidate,
    const CandidateMetrics& metrics,
    std::int64_t iterations,
    std::int64_t stepTrials,
    int restarts,
    double solveTime
) {
    PdlpResult result;
    result.status = status;
    result.statusMessage = std::move(message);
    result.primal = std::move(candidate.primal);
    result.rowDual = std::move(candidate.dual);
    result.primalObjective = metrics.primalObjective;
    result.dualObjective = metrics.dualObjective;
    result.primalResidual = metrics.primalResidual;
    result.dualResidual = metrics.dualResidual;
    result.relativeGap = metrics.relativeGap;
    result.iterations = iterations;
    result.stepTrials = stepTrials;
    result.restartCount = restarts;
    result.solveTimeSeconds = solveTime;
    return result;
}

void difference(
    const std::vector<double>& left,
    const std::vector<double>& right,
    std::vector<double>& result
) {
    result.resize(left.size());
    for (std::size_t i = 0; i < left.size(); ++i) {
        result[i] = left[i] - right[i];
    }
}

double squaredDistance(
    const std::vector<double>& left,
    const std::vector<double>& right
) {
    double sum = 0.0;
    for (std::size_t i = 0; i < left.size(); ++i) {
        const double difference = left[i] - right[i];
        sum += difference * difference;
    }
    return sum;
}

PdlpResult iterateToCompletion(
    const CompiledLp& problem,
    const PdlpOptions& options,
    const CompiledLp& working,
    const ProblemScaling* scalingPointer,
    const DiagonalPreconditioner& preconditioner,
    double maximumSafeGlobalStep,
    IterationBackend& iterate,
    const std::string& backendMessage,
    const std::vector<double>& initialPrimal,
    const std::vector<double>& initialDual,
    Executor* executor,
    const SpmvPlan* plan,
    const Clock::time_point& start
);

PdlpResult invalidProblem(std::string message, double solveTime) {
    PdlpResult result;
    result.status = PdlpStatus::InvalidProblem;
    result.statusMessage = std::move(message);
    result.solveTimeSeconds = solveTime;
    return result;
}

}  // namespace

const char* toString(PdlpStatus status) noexcept {
    switch (status) {
        case PdlpStatus::Optimal: return "optimal";
        case PdlpStatus::Infeasible: return "infeasible";
        case PdlpStatus::Unbounded: return "unbounded";
        case PdlpStatus::IterationLimit: return "iteration_limit";
        case PdlpStatus::TimeLimit: return "time_limit";
        case PdlpStatus::NumericalFailure: return "numerical_failure";
        case PdlpStatus::InvalidProblem: return "invalid_problem";
    }
    return "unknown";
}

PdlpResult PdlpSolver::solve(
    const CompiledLp& problem,
    const PdlpOptions& options
) const {
    const Clock::time_point start = Clock::now();
    try {
        problem.validate();
    } catch (const std::exception& error) {
        return invalidProblem(error.what(), elapsedSeconds(start));
    }

    if (options.iterationLimit < 0 || options.terminationCheckFrequency <= 0 ||
        options.primalTolerance < 0.0 || options.dualTolerance < 0.0 ||
        options.gapTolerance < 0.0 || options.initialStepSafety <= 0.0 ||
        options.minimumPrimalWeight <= 0.0 ||
        options.maximumPrimalWeight < options.minimumPrimalWeight ||
        options.cudaDevice < 0) {
        return invalidProblem("Invalid PDLP options", elapsedSeconds(start));
    }

    // An explicit CUDA request that cannot be honoured fails here, before any
    // setup work, and says why. It is never downgraded to a CPU solve.
    if (options.backend == ComputeBackend::Cuda) {
        const CudaAvailability availability = cudaAvailability(options.cudaDevice);
        if (!availability.usable) {
            return invalidProblem(
                "CUDA backend requested but unavailable: " + availability.reason,
                elapsedSeconds(start));
        }
    }

    const int columns = problem.numColumns();
    const int rows = problem.numRows();

    // Threading is worth its barriers only above a work threshold; small models
    // stay strictly serial, which also keeps the unit tests deterministic.
    std::unique_ptr<Executor> executor;
    std::unique_ptr<SpmvPlan> plan;
    const bool parallelWorthwhile =
        options.threadCount != 1 &&
        static_cast<std::int64_t>(problem.matrix.nonzeros()) >=
            options.parallelNonzeroThreshold;
    if (parallelWorthwhile) {
        auto candidateExecutor = std::make_unique<Executor>(options.threadCount);
        if (candidateExecutor->threadCount() > 1) {
            plan = std::make_unique<SpmvPlan>(
                SpmvPlan::build(problem.matrix, candidateExecutor->threadCount()));
            // The scaled matrix shares the sparsity pattern, so one plan serves
            // both.
            executor = std::move(candidateExecutor);
        }
    }

    // Equilibrate before solving. The iteration runs on `working`; termination
    // is always evaluated against `problem`, so reported residuals and the
    // returned solution stay in the caller's units.
    std::unique_ptr<ProblemScaling> scaling;
    if (options.useRuizScaling && options.ruizIterations > 0 &&
        problem.matrix.nonzeros() > 0) {
        scaling = std::make_unique<ProblemScaling>(
            RuizScaler::equilibrate(problem, options.ruizIterations));
    }
    const CompiledLp& working = scaling ? scaling->problem : problem;

    std::vector<double> initialPrimal(static_cast<std::size_t>(columns));
    const std::vector<double> initialDual(static_cast<std::size_t>(rows), 0.0);
    for (int column = 0; column < columns; ++column) {
        initialPrimal[static_cast<std::size_t>(column)] = initialValue(
            working.variableLower[static_cast<std::size_t>(column)],
            working.variableUpper[static_cast<std::size_t>(column)]
        );
    }

    const DiagonalPreconditioner preconditioner = Preconditioner::compute(
        working,
        options.useDiagonalPreconditioning
    );

    // With the Pock-Chambolle scaling above, any global step <= 1 satisfies the
    // PDHG convergence condition, so no spectral estimate is required. Without
    // it the bound is 1/||A||, and the power iteration approaches ||A|| from
    // below, so an extra margin is applied and widened further if the iteration
    // did not converge.
    double maximumSafeGlobalStep = 0.99;
    if (!options.useDiagonalPreconditioning) {
        bool converged = false;
        const double norm = Preconditioner::estimateSpectralNorm(
            working.matrix,
            options.powerIterationCount,
            &converged
        );
        if (norm > 0.0) {
            const double margin = converged ? 1.02 : 1.25;
            maximumSafeGlobalStep = 0.99 / (norm * margin);
        } else {
            maximumSafeGlobalStep = 1.0;
        }
    }

    // Backend selection. Deterministic: it depends only on the options, the
    // build, the device, and the size of the working matrix. The executor
    // stays in use on every backend, because termination and certificate
    // checks always run on the host.
    std::unique_ptr<IterationBackend> iterate;
    std::string backendMessage;
    {
        const auto workingNonzeros =
            static_cast<std::int64_t>(working.matrix.nonzeros());
        bool wantCuda = false;
        switch (options.backend) {
            case ComputeBackend::Cpu:
                backendMessage = "cpu: requested";
                break;
            case ComputeBackend::Cuda:
                wantCuda = true;
                break;
            case ComputeBackend::Auto:
                if (workingNonzeros < options.cudaNonzeroThreshold) {
                    backendMessage = "auto -> cpu: " + std::to_string(workingNonzeros) +
                        " nonzeros is below cudaNonzeroThreshold (" +
                        std::to_string(options.cudaNonzeroThreshold) + ")";
                } else {
                    const CudaAvailability availability =
                        cudaAvailability(options.cudaDevice);
                    if (availability.usable) {
                        wantCuda = true;
                    } else {
                        backendMessage = "auto -> cpu: " + availability.reason;
                    }
                }
                break;
        }

        if (wantCuda) {
            CudaBackendConfig config;
            config.device = options.cudaDevice;
            std::string error;
            iterate = makeCudaIterationBackend(working, preconditioner, config, error);
            if (!iterate) {
                if (options.backend == ComputeBackend::Cuda) {
                    return invalidProblem(
                        "CUDA backend requested but could not be started: " + error,
                        elapsedSeconds(start));
                }
                backendMessage = "auto -> cpu: CUDA backend could not be started: " + error;
            } else {
                backendMessage = options.backend == ComputeBackend::Cuda
                    ? "cuda: requested, device " + std::to_string(options.cudaDevice)
                    : "auto -> cuda: " + std::to_string(workingNonzeros) +
                          " nonzeros, device " + std::to_string(options.cudaDevice);
            }
        }
        if (!iterate) {
            iterate = makeCpuIterationBackend(
                working, preconditioner, executor.get(), plan.get());
        }
    }

    // A device failure part-way through a solve is reported as a failed solve
    // rather than propagated. The CPU backend's behaviour is unchanged: its
    // exceptions still propagate exactly as they did before backends existed.
    try {
        return iterateToCompletion(
            problem, options, working, scaling.get(), preconditioner,
            maximumSafeGlobalStep, *iterate, backendMessage,
            initialPrimal, initialDual, executor.get(), plan.get(), start);
    } catch (const std::exception& error) {
        if (iterate->kind() != ComputeBackend::Cuda) {
            throw;
        }
        PdlpResult result;
        result.status = PdlpStatus::NumericalFailure;
        result.statusMessage = std::string("CUDA backend error: ") + error.what();
        result.executedBackend = ComputeBackend::Cuda;
        result.backendMessage = backendMessage;
        result.backendProfile = iterate->profile();
        result.solveTimeSeconds = elapsedSeconds(start);
        return result;
    }
}

namespace {

// The PDLP iteration proper -- linesearch, averaging, termination,
// certificates, restarts and polishing -- written once against
// IterationBackend, so every backend runs the same algorithm.
PdlpResult iterateToCompletion(
    const CompiledLp& problem,
    const PdlpOptions& options,
    const CompiledLp& working,
    const ProblemScaling* scalingPointer,
    const DiagonalPreconditioner& preconditioner,
    double maximumSafeGlobalStep,
    IterationBackend& iterate,
    const std::string& backendMessage,
    const std::vector<double>& initialPrimal,
    const std::vector<double>& initialDual,
    Executor* executor,
    const SpmvPlan* plan,
    const Clock::time_point& start
) {
    StepController stepController(options, maximumSafeGlobalStep);
    stepController.setPrimalWeight(options.initialPrimalWeight);
    iterate.reset(initialPrimal, initialDual);
    TerminationChecker checker(options, problem, executor, plan);
    RestartController restartController(options);
    InfeasibilityDetector detector(problem, options, executor, plan);

    // Candidate ray directions, in the original problem's coordinates.
    std::vector<double> primalDirection;
    std::vector<double> dualDirection;

    // Scores a working-coordinate iterate against the original problem. The
    // original-coordinate vectors stay addressable afterwards, so an improved
    // candidate is kept without converting twice.
    std::vector<double> originalPrimal;
    std::vector<double> originalDual;
    const std::vector<double>* scoredPrimal = nullptr;
    const std::vector<double>* scoredDual = nullptr;
    double hostCheckSeconds = 0.0;

    const auto score = [&](const std::vector<double>& primal,
                           const std::vector<double>& dual) {
        if (scalingPointer != nullptr) {
            scalingPointer->toOriginal(primal, dual, originalPrimal, originalDual);
            scoredPrimal = &originalPrimal;
            scoredDual = &originalDual;
        } else {
            scoredPrimal = &primal;
            scoredDual = &dual;
        }
        return checker.evaluate(*scoredPrimal, *scoredDual);
    };

    CandidateMetrics currentMetrics = score(iterate.primal(), iterate.dual());
    CandidateIterate best{*scoredPrimal, *scoredDual};
    CandidateMetrics bestMetrics = currentMetrics;
    double restartBaseline = currentMetrics.kktScore;
    int iterationsSinceRestart = 0;
    int restartCount = 0;

    // Anchor for the primal weight update: the iterate at the last restart, in
    // working coordinates.
    std::vector<double> restartPrimal = iterate.primal();
    std::vector<double> restartDual = iterate.dual();
    std::int64_t stepTrials = 0;


    // Backend identity and accounting, stamped onto every exit path.
    const auto withBackend = [&](PdlpResult value) {
        value.executedBackend = iterate.kind();
        value.backendMessage = backendMessage;
        value.backendProfile = iterate.profile();
        value.hostCheckSeconds = hostCheckSeconds;
        return value;
    };

    if (checker.isOptimal(currentMetrics)) {
        return withBackend(makeResult(
            PdlpStatus::Optimal,
            "Initial point satisfies the requested tolerances",
            std::move(best),
            currentMetrics,
            0,
            0,
            0,
            elapsedSeconds(start)
        ));
    }

    // Stamped onto every exit path below.
    const auto withDiagnostics = [&](PdlpResult value) {
        value.finalStepSize = stepController.parameters().globalStep;
        value.staticStepBound = stepController.parameters().maximumSafeGlobalStep;
        value.finalPrimalWeight = stepController.parameters().primalWeight;
        return withBackend(std::move(value));
    };

    PdlpStatus limitStatus = PdlpStatus::IterationLimit;
    std::string limitMessage = "Iteration limit reached";

    for (std::int64_t iteration = 0; iteration < options.iterationLimit; ++iteration) {
        // Short-circuits to a single predictable branch when no limit is set,
        // which is the default, so the clock is never read in that case.
        if (options.timeLimitSeconds > 0.0 &&
            elapsedSeconds(start) >= options.timeLimitSeconds) {
            limitStatus = PdlpStatus::TimeLimit;
            limitMessage = "Time limit reached";
            break;
        }

        // Adaptive linesearch. A rejected trial costs one pass over the matrix
        // and is retried with a smaller step; eta_bar is a property of the local
        // curvature rather than of the current step, so the loop converges in a
        // trial or two rather than by repeated halving.
        bool committed = false;
        for (int attempt = 0; attempt < std::max(options.maximumStepTrials, 1); ++attempt) {
            const KernelTrialResult trialResult =
                iterate.trial(stepController.parameters());
            ++stepTrials;
            if (!trialResult.finite) {
                return withDiagnostics(makeResult(
                    PdlpStatus::NumericalFailure,
                    "PDHG generated a non-finite iterate",
                    std::move(best),
                    bestMetrics,
                    iterate.iteration(),
                    stepTrials,
                    restartCount,
                    elapsedSeconds(start)
                ));
            }

            const bool accept = !options.useAdaptiveLinesearch ||
                stepController.evaluateTrial(
                    trialResult.primalMovementWeighted,
                    trialResult.dualMovementWeighted,
                    trialResult.interaction,
                    iterate.iteration()
                );
            if (accept) {
                iterate.commit();
                committed = true;
                break;
            }
        }
        if (!committed) {
            // The linesearch never accepted within its budget. Committing the
            // last trial anyway would apply a step the curvature test rejected.
            return withDiagnostics(makeResult(
                PdlpStatus::NumericalFailure,
                "Adaptive linesearch failed to accept a step",
                std::move(best),
                bestMetrics,
                iterate.iteration(),
                stepTrials,
                restartCount,
                elapsedSeconds(start)
            ));
        }

        ++iterationsSinceRestart;

        if (options.useAveraging) {
            iterate.addToAverage(stepController.parameters().globalStep);
        }

        if (iterate.iteration() % options.terminationCheckFrequency != 0) {
            continue;
        }

        const Clock::time_point checkStart = Clock::now();
        currentMetrics = score(iterate.primal(), iterate.dual());
        if (currentMetrics.kktScore < bestMetrics.kktScore) {
            best.primal = *scoredPrimal;
            best.dual = *scoredDual;
            bestMetrics = currentMetrics;
        }

        CandidateMetrics averagedMetrics = currentMetrics;
        const bool haveAverage = options.useAveraging && !iterate.averageEmpty();
        if (haveAverage) {
            const CandidateIterate& averaged = iterate.average();
            averagedMetrics = score(averaged.primal, averaged.dual);
            if (averagedMetrics.kktScore < bestMetrics.kktScore) {
                best.primal = *scoredPrimal;
                best.dual = *scoredDual;
                bestMetrics = averagedMetrics;
            }
        }

        // Certificate test. The iterate difference since the last restart is the
        // direction the sequence is diverging along; on an infeasible or
        // unbounded problem it converges to a ray. Testing it costs two matrix
        // passes, so it happens only on a check boundary.
        if (options.detectInfeasibility) {
            difference(iterate.primal(), restartPrimal, primalDirection);
            difference(iterate.dual(), restartDual, dualDirection);

            std::vector<double> rowRay;
            std::vector<double> columnRay;
            if (scalingPointer != nullptr) {
                scalingPointer->toOriginal(
                    primalDirection, dualDirection, columnRay, rowRay);
            } else {
                columnRay = primalDirection;
                rowRay = dualDirection;
            }

            const InfeasibilityVerdict dualVerdict = detector.testDualRay(rowRay);
            if (dualVerdict.provesPrimalInfeasible) {
                PdlpResult result = withDiagnostics(makeResult(
                    PdlpStatus::Infeasible,
                    "Primal infeasible: Farkas ray found",
                    std::move(best),
                    bestMetrics,
                    iterate.iteration(),
                    stepTrials,
                    restartCount,
                    elapsedSeconds(start)
                ));
                result.dualRay = std::move(rowRay);
                return result;
            }

            const InfeasibilityVerdict primalVerdict = detector.testPrimalRay(columnRay);
            if (primalVerdict.provesUnbounded) {
                PdlpResult result = withDiagnostics(makeResult(
                    PdlpStatus::Unbounded,
                    "Unbounded: improving ray found",
                    std::move(best),
                    bestMetrics,
                    iterate.iteration(),
                    stepTrials,
                    restartCount,
                    elapsedSeconds(start)
                ));
                result.primalRay = std::move(columnRay);
                return result;
            }
        }

        if (checker.isOptimal(bestMetrics)) {
            return withDiagnostics(makeResult(
                PdlpStatus::Optimal,
                "Primal, dual, and gap tolerances satisfied",
                std::move(best),
                bestMetrics,
                iterate.iteration(),
                stepTrials,
                restartCount,
                elapsedSeconds(start)
            ));
        }

        const RestartDecision restart = restartController.choose(
            currentMetrics,
            averagedMetrics,
            restartBaseline,
            iterationsSinceRestart,
            iterate.iteration()
        );
        if (restart.choice != RestartChoice::None) {
            const bool fromAverage =
                restart.choice == RestartChoice::Average && haveAverage;
            const std::vector<double>* selectedPrimal = &iterate.primal();
            const std::vector<double>* selectedDual = &iterate.dual();
            if (fromAverage) {
                const CandidateIterate& averaged = iterate.average();
                selectedPrimal = &averaged.primal;
                selectedDual = &averaged.dual;
            }

            // Straight-line displacement over the cycle that just ended.
            stepController.updatePrimalWeight(
                std::sqrt(squaredDistance(*selectedPrimal, restartPrimal)),
                std::sqrt(squaredDistance(*selectedDual, restartDual))
            );

            if (fromAverage) {
                // Restarting from the current iterate needs no copy: the state
                // already holds it, and its A*x is already correct.
                iterate.restartFromAverage();
            }

            restartPrimal = iterate.primal();
            restartDual = iterate.dual();
            restartBaseline = restart.candidateScore;
            iterate.resetAverage();
            restartController.reset();
            iterationsSinceRestart = 0;
            ++restartCount;
        }
        hostCheckSeconds += elapsedSeconds(checkStart);
    }

    // Polishing reuses the backend and restarts its iteration counter, so the
    // main loop's count is captured first and used by every exit below.
    const std::int64_t mainIterations = iterate.iteration();

    if (options.useFeasibilityPolishing && !best.primal.empty()) {
        double remaining = 0.0;
        if (options.timeLimitSeconds > 0.0) {
            remaining = options.timeLimitSeconds - elapsedSeconds(start);
            if (remaining <= 0.0) {
                return withDiagnostics(makeResult(
                    limitStatus,
                    limitMessage,
                    std::move(best),
                    bestMetrics,
                    mainIterations,
                    stepTrials,
                    restartCount,
                    elapsedSeconds(start)
                ));
            }
        }

        PolishingInput input;
        input.working = &working;
        input.scaling = scalingPointer;
        input.preconditioner = &preconditioner;
        input.originalStart = best;
        if (scalingPointer != nullptr) {
            scalingPointer->toScaled(
                best.primal,
                best.dual,
                input.scaledStart.primal,
                input.scaledStart.dual
            );
        } else {
            input.scaledStart = best;
        }
        input.startMetrics = bestMetrics;
        input.steps = stepController.parameters();
        input.remainingSeconds = remaining;
        input.executor = executor;
        input.plan = plan;
        // Polishing reuses the solve's backend -- and on a device, the problem
        // already resident there -- rather than building a second one. The
        // main iteration is finished, so its state may be overwritten.
        input.backend = &iterate;

        PolishingResult polished =
            FeasibilityPolisher::polish(options, input, checker);
        hostCheckSeconds += polished.hostCheckSeconds;
        if (polished.metrics.kktScore < bestMetrics.kktScore) {
            best = std::move(polished.candidate);
            bestMetrics = polished.metrics;
        }
        if (checker.isOptimal(bestMetrics)) {
            return withDiagnostics(makeResult(
                PdlpStatus::Optimal,
                "Tolerances satisfied after refinement polishing",
                std::move(best),
                bestMetrics,
                mainIterations + polished.iterations,
                stepTrials + polished.stepTrials,
                restartCount,
                elapsedSeconds(start)
            ));
        }
    }

    return withDiagnostics(makeResult(
        limitStatus,
        limitMessage,
        std::move(best),
        bestMetrics,
        mainIterations,
        stepTrials,
        restartCount,
        elapsedSeconds(start)
    ));
}

}  // namespace

}  // namespace pdlp
