#include "solver/nlp.h"
#include <limits>
#include <stdexcept>

namespace solver {
Classification classify(const nlp::Problem& problem) {
    Classification result;
    result.problemClass = ProblemClass::NLP;
    result.hints.numColumns = problem.variableBounds().size();
    result.hints.numRows = problem.constraintBounds().size();
    if (result.hints.numColumns > static_cast<size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("NLP variable count exceeds supported indices");
    result.hints.numContinuous = static_cast<int>(result.hints.numColumns);
    result.hints.numNonzeros = -1;
    // A callback's Jacobian pattern need not be known before evaluation.
    // Do not invoke user evaluations merely to obtain structural hints.
    return result;
}

DispatchDecision dispatch(const nlp::Problem&, std::optional<Engine> requested) {
    if (requested && *requested != Engine::Nlp)
        return {Engine::Unsupported, "a nonlinear model requires the NLP engine"};
    return {Engine::Nlp, "smooth nonlinear model: elastic SQP; affine presolve/postsolve bypassed"};
}

NlpSolveResult solve(const nlp::Problem& problem, const std::vector<double>& initial,
                     const nlp::Options& options, std::optional<Engine> requested) {
    NlpSolveResult result;
    try {
        result.classification = classify(problem);
        auto decision = dispatch(problem, requested);
        result.engine = decision.engine;
        result.engineReason = decision.reason;
        if (decision.engine != Engine::Nlp) {
            result.status = nlp::Status::InvalidProblem;
            result.message = decision.reason;
            return result;
        }
        result.executedEngine = Engine::Nlp;
        static_cast<nlp::Result&>(result) = nlp::Solver().solve(problem, initial, options);
    } catch (const std::exception& e) {
        result.status = nlp::Status::InvalidProblem;
        result.message = e.what();
    }
    return result;
}
} // namespace solver
