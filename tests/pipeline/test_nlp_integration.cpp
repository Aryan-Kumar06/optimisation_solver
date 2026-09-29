#include "solver/orchestrator.h"
#include <cmath>
#include <stdexcept>
#include <string>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}
int main() {
    using namespace nlp;
    auto x = variable(0), y = variable(1);
    Model problem({{}, {}}, -x-y, {square(x)+square(y)}, {{-infinity, 1}});
    auto classification = solver::classify(problem);
    require(classification.problemClass == solver::ProblemClass::NLP, "NLP classification");
    require(classification.hints.numColumns == 2 && classification.hints.numRows == 1, "NLP dimensions");
    require(classification.hints.numNonzeros == -1, "unknown Jacobian pattern");
    require(classification.hints.numContinuous == 2, "continuous classification");
    require(std::string(solver::toString(classification.problemClass)) == "NLP", "class label");
    require(solver::parseEngine("nlp") == solver::Engine::Nlp, "engine name");
    require(solver::parseEngine("nlp_sqp") == solver::Engine::Nlp, "engine alias");
    require(solver::dispatch(problem).engine == solver::Engine::Nlp, "automatic dispatch");
    require(solver::dispatch(problem, solver::Engine::Qp).engine == solver::Engine::Unsupported, "invalid force dispatch");
    auto result = solver::solve(problem, {0.2, 0.1});
    require(result.engine == solver::Engine::Nlp && result.executedEngine == solver::Engine::Nlp, "executed engine metadata");
    require(!result.engineReason.empty(), "missing engine reason");
    require(result.status == Status::FirstOrderStationary && result.feasible, "NLP solve");
    require(std::abs(result.objective + std::sqrt(2.0)) < 1e-6, "original objective");
    require(result.constraintMultipliers.size() == 1 && result.constraintMultipliers[0] > 0, "KKT multiplier sign");
    auto refused = solver::solve(problem, {0.2, 0.1}, {}, solver::Engine::Pdlp);
    require(refused.status == Status::InvalidProblem && refused.executedEngine == solver::Engine::Unsupported, "forced affine engine must not execute");
    Options options; options.iterationLimit = 0;
    auto limited = solver::solve(problem, {0.2, 0.1}, options);
    require(limited.status == Status::IterationLimit && limited.hasPrimal, "budget preservation");
    options.callback = [](const Iteration&) { return false; };
    require(solver::solve(problem, {0.2, 0.1}, options).status == Status::UserStopped, "callback preservation");

    // Affine models cannot be silently solved by an incompatible forced engine,
    // even when affine presolve could otherwise eliminate the whole model.
    model::Model affine;
    affine.variables.push_back({"x", model::VariableType::Continuous, 0, 1});
    affine.objective.linearTerms.push_back({0, 1});
    solver::SolverOptions force; force.forceEngine = solver::Engine::Nlp;
    auto wrong = solver::solve(affine, force);
    require(wrong.status == solver::SolveStatus::Unsupported && wrong.executedEngine == solver::Engine::Unsupported, "affine/NLP model boundary");
    auto reduced = solver::solveReduced(affine, solver::classify(affine), force);
    require(reduced.status == solver::SolveStatus::Unsupported, "reduced affine/NLP boundary");
    auto original = solver::solve(affine);
    require(original.status == solver::SolveStatus::Optimal, "affine regression");
}
