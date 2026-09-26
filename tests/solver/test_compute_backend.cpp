// Compute-backend selection through the public pipeline: the request reaches
// the engine, the engine records what actually ran, and an explicit CUDA
// request is never answered by a silent CPU solve. Runs on every build; the
// CUDA-unavailable branches are the ones a CPU-only build exercises.

#include "pdlp/compute_backend.h"
#include "qp/compute_backend.h"
#include "solver/orchestrator.h"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

namespace {

const double INF = std::numeric_limits<double>::infinity();
int checks = 0;
int failures = 0;

void ck(bool ok, const std::string& what) {
    ++checks;
    if (!ok) {
        ++failures;
        std::printf("    FAIL  %s\n", what.c_str());
    }
}

// min -3x - 5y  s.t.  x <= 4, 2y <= 12, 3x + 2y <= 18, x, y >= 0.
// Optimum -36 at (2, 6).
model::Model textbookLp() {
    model::Model m;
    for (const char* name : {"x", "y"}) {
        model::Variable v;
        v.name = name;
        v.lowerBound = 0.0;
        v.upperBound = INF;
        m.variables.push_back(v);
    }
    const auto row = [&](const char* name, double upper, std::vector<model::LinearTerm> terms) {
        model::Constraint c;
        c.name = name;
        c.lowerBound = -INF;
        c.upperBound = upper;
        c.linearTerms = std::move(terms);
        m.constraints.push_back(c);
    };
    row("a", 4.0, {{0, 1.0}});
    row("b", 12.0, {{1, 2.0}});
    row("c", 18.0, {{0, 3.0}, {1, 2.0}});
    m.objective.sense = model::ObjectiveSense::Minimize;
    m.objective.linearTerms = {{0, -3.0}, {1, -5.0}};
    return m;
}

model::Model textbookQp() {
    model::Model m = textbookLp();
    m.objective.quadraticTerms = {{0, 0, 1.0}, {1, 1, 1.0}};
    return m;
}

void testParseAndName() {
    ck(solver::parseComputeBackend("auto") == solver::ComputeBackend::Auto, "parse auto");
    ck(solver::parseComputeBackend("cpu") == solver::ComputeBackend::Cpu, "parse cpu");
    ck(solver::parseComputeBackend("cuda") == solver::ComputeBackend::Cuda, "parse cuda");
    ck(!solver::parseComputeBackend("gpu").has_value(), "unknown backend names are rejected");
    ck(std::string(solver::toString(solver::ComputeBackend::Cuda)) == "cuda", "backend name");
    const solver::SolverOptions defaults;
    ck(defaults.backend == solver::ComputeBackend::Auto && defaults.cudaDevice == 0,
       "defaults are Auto on device 0");
}

void testDefaultRunsOnCpuAndSaysWhy() {
    solver::SolverOptions options;
    options.forceEngine = solver::Engine::Pdlp;
    const solver::SolveResult r = solver::solve(textbookLp(), options);
    ck(r.status == solver::SolveStatus::Optimal, "PDLP solves the textbook LP");
    ck(std::abs(r.objectiveValue + 36.0) < 1e-4, "PDLP objective");
    ck(r.executedBackend == solver::ComputeBackend::Cpu, "a small LP runs on the CPU");
    ck(r.backendReason.find("below cudaNonzeroThreshold") != std::string::npos,
       "the reason for the CPU choice is recorded: " + r.backendReason);
}

void testForcedCpu() {
    solver::SolverOptions options;
    options.forceEngine = solver::Engine::Pdlp;
    options.backend = solver::ComputeBackend::Cpu;
    const solver::SolveResult r = solver::solve(textbookLp(), options);
    ck(r.status == solver::SolveStatus::Optimal, "forced CPU solves");
    ck(r.executedBackend == solver::ComputeBackend::Cpu, "forced CPU runs on the CPU");
    ck(r.backendReason == "cpu: requested", "forced CPU reason: " + r.backendReason);

    solver::SolverOptions qpOptions;
    qpOptions.backend = solver::ComputeBackend::Cpu;
    const solver::SolveResult q = solver::solve(textbookQp(), qpOptions);
    ck(q.executedEngine == solver::Engine::Qp, "a quadratic objective routes to the QP engine");
    ck(q.executedBackend == solver::ComputeBackend::Cpu, "QP forced CPU runs on the CPU");
}

// An explicit CUDA request on a machine that cannot honour it is Unsupported,
// with the reason, and no engine runs.
void testExplicitCudaIsNeverSilentlyCpu() {
    if (!pdlp::cudaAvailability(0).usable) {
        solver::SolverOptions options;
        options.forceEngine = solver::Engine::Pdlp;
        options.backend = solver::ComputeBackend::Cuda;
        const solver::SolveResult r = solver::solve(textbookLp(), options);
        ck(r.status == solver::SolveStatus::Unsupported, "PDLP + unavailable CUDA is Unsupported");
        ck(r.message.find("CUDA backend requested but unavailable") != std::string::npos,
           "the message names the CUDA request: " + r.message);
        ck(!r.hasPrimal, "no solution is reported");
        ck(r.executedEngine == solver::Engine::Unsupported, "no engine was invoked");
    }
    if (!qp::cudaAvailability(0).usable) {
        solver::SolverOptions options;
        options.backend = solver::ComputeBackend::Cuda;
        const solver::SolveResult r = solver::solve(textbookQp(), options);
        ck(r.status == solver::SolveStatus::Unsupported, "QP + unavailable CUDA is Unsupported");
        ck(!r.hasPrimal, "no QP solution is reported");
    }
}

// Engines without a CUDA backend run on the CPU whatever was requested, and
// say so.
void testEnginesWithoutCudaSayCpu() {
    solver::SolverOptions options;
    options.forceEngine = solver::Engine::DualSimplex;
    options.backend = solver::ComputeBackend::Cuda;
    const solver::SolveResult r = solver::solve(textbookLp(), options);
    ck(r.status == solver::SolveStatus::Optimal, "dual simplex still solves");
    ck(r.executedBackend == solver::ComputeBackend::Cpu, "dual simplex runs on the CPU");
    ck(r.backendReason.find("has no CUDA backend") != std::string::npos,
       "the CUDA request is acknowledged as not applicable: " + r.backendReason);
}

}  // namespace

int main() {
    testParseAndName();
    testDefaultRunsOnCpuAndSaysWhy();
    testForcedCpu();
    testExplicitCudaIsNeverSilentlyCpu();
    testEnginesWithoutCudaSayCpu();
    std::printf("compute backend: %d checks, %d failures\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
