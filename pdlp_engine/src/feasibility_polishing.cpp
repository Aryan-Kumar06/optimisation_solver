#include "pdlp/feasibility_polishing.h"

#include "pdlp/iteration_backend.h"
#include "pdlp/step_controller.h"

#include <algorithm>
#include <chrono>
#include <memory>
#include <vector>

namespace pdlp {

PolishingResult FeasibilityPolisher::polish(
    const PdlpOptions& options,
    const PolishingInput& input,
    TerminationChecker& checker
) {
    PolishingResult result{input.originalStart, input.startMetrics, 0};
    if (!options.useFeasibilityPolishing || options.polishingIterations <= 0) {
        return result;
    }

    const CompiledLp& working = *input.working;
    const ProblemScaling* scaling = input.scaling;

    // This CPU v1 uses a conservative refinement phase: keep the original
    // objective, halve the step, average the refined iterates, and retain only
    // candidates with a better complete KKT score. A production feasibility
    // polishing implementation can replace this component without changing the
    // solver interface.
    StepParameters steps = input.steps;
    steps.globalStep *= 0.5;

    std::unique_ptr<IterationBackend> ownedBackend;
    IterationBackend* backend = input.backend;
    if (backend == nullptr) {
        ownedBackend = makeCpuIterationBackend(
            working, *input.preconditioner, input.executor, input.plan);
        backend = ownedBackend.get();
    }
    IterationBackend& iterate = *backend;
    iterate.reset(input.scaledStart.primal, input.scaledStart.dual);

    StepController stepController(options, steps.maximumSafeGlobalStep);
    stepController.setPrimalWeight(steps.primalWeight);

    std::vector<double> originalPrimal;
    std::vector<double> originalDual;
    const std::vector<double>* scoredPrimal = nullptr;
    const std::vector<double>* scoredDual = nullptr;

    // Scores a working-coordinate iterate against the original problem, leaving
    // the original-coordinate vectors addressable so an improvement can be kept
    // without a second conversion.
    const auto score = [&](const std::vector<double>& primal,
                           const std::vector<double>& dual) {
        if (scaling != nullptr) {
            scaling->toOriginal(primal, dual, originalPrimal, originalDual);
            scoredPrimal = &originalPrimal;
            scoredDual = &originalDual;
        } else {
            scoredPrimal = &primal;
            scoredDual = &dual;
        }
        return checker.evaluate(*scoredPrimal, *scoredDual);
    };

    const auto start = std::chrono::steady_clock::now();
    const int checkFrequency = std::max(options.terminationCheckFrequency, 1);

    for (int iteration = 0; iteration < options.polishingIterations; ++iteration) {
        bool committed = false;
        bool broken = false;
        for (int attempt = 0; attempt < std::max(options.maximumStepTrials, 1); ++attempt) {
            const KernelTrialResult trialResult = iterate.trial(steps);
            ++result.stepTrials;
            if (!trialResult.finite) {
                broken = true;
                break;
            }
            if (!options.useAdaptiveLinesearch) {
                iterate.commit();
                committed = true;
                break;
            }
            const bool accept = stepController.evaluateTrial(
                trialResult.primalMovementWeighted,
                trialResult.dualMovementWeighted,
                trialResult.interaction,
                iterate.iteration()
            );
            steps.globalStep = stepController.parameters().globalStep;
            if (accept) {
                iterate.commit();
                committed = true;
                break;
            }
        }
        if (broken || !committed) {
            break;
        }

        if (options.useAveraging) {
            iterate.addToAverage(steps.globalStep);
        }
        ++result.iterations;

        const bool lastIteration = iteration + 1 == options.polishingIterations;
        if ((iteration + 1) % checkFrequency != 0 && !lastIteration) {
            continue;
        }

        // Polishing respects the solve's remaining budget; it previously could
        // run its full iteration count after a time limit had already expired.
        if (input.remainingSeconds > 0.0) {
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - start).count();
            if (elapsed >= input.remainingSeconds) {
                break;
            }
        }

        const auto checkStart = std::chrono::steady_clock::now();
        const CandidateMetrics currentMetrics = score(iterate.primal(), iterate.dual());
        if (currentMetrics.kktScore < result.metrics.kktScore) {
            result.candidate.primal = *scoredPrimal;
            result.candidate.dual = *scoredDual;
            result.metrics = currentMetrics;
        }

        if (options.useAveraging && !iterate.averageEmpty()) {
            const CandidateIterate& averaged = iterate.average();
            const CandidateMetrics averagedMetrics = score(averaged.primal, averaged.dual);
            if (averagedMetrics.kktScore < result.metrics.kktScore) {
                result.candidate.primal = *scoredPrimal;
                result.candidate.dual = *scoredDual;
                result.metrics = averagedMetrics;
            }
        }

        result.hostCheckSeconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - checkStart).count();

        if (checker.isOptimal(result.metrics)) {
            break;
        }
    }

    return result;
}

}  // namespace pdlp
