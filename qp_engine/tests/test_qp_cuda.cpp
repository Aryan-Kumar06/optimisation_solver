// CPU-vs-CUDA equivalence for the hybrid ADMM backend. Built only when
// QP_ENABLE_CUDA is on; exits 77 (CTest: skipped) without a usable GPU.

#include "admm_backend_contract.h"

#include "qp/admm_backend.h"
#include "qp/compute_backend.h"
#include "qp/qp_solver.h"

#include <cstdio>
#include <cstdlib>
#include <functional>
#include <string>

namespace {

int failures = 0;

void run(const std::string& name, const std::function<void()>& test) {
    try {
        test();
        std::printf("  pass  %s\n", name.c_str());
    } catch (const std::exception& error) {
        std::printf("  FAIL  %s: %s\n", name.c_str(), error.what());
        ++failures;
    }
}

// cuSPARSE and the fused reductions sum in a different order from the CPU;
// nothing else may differ.
constexpr double kTolerance = 1e-10;

void backendEquivalence() {
    const qpcontract::BackendFactory cpu = [](const qp::QpModel& model) {
        return qp::makeCpuAdmmBackend(model, nullptr, nullptr, nullptr);
    };
    const qpcontract::BackendFactory cuda = [](const qp::QpModel& model) {
        std::string error;
        auto backend = qp::makeCudaAdmmBackend(model, 0, error);
        if (!backend) {
            throw std::runtime_error("CUDA backend creation failed: " + error);
        }
        return backend;
    };
    for (const qpcontract::Fixture& fixture : qpcontract::fixtures()) {
        run("backend " + fixture.name, [&] {
            qpcontract::compareBackends(fixture, cpu, cuda, kTolerance);
        });
    }
}

qp::QpModel infeasibleModel() {
    qp::QpModel m;
    m.P = qp::SparseMatrix::fromTriplets(2, 2, {0, 1}, {0, 1}, {1.0, 1.0});
    m.A = qp::SparseMatrix::fromTriplets(2, 2, {0, 0, 1, 1}, {0, 1, 0, 1}, {1, 1, 1, 1});
    m.q = {0.0, 0.0};
    m.l = {-qpcontract::kInf, 3.0};
    m.u = {1.0, qpcontract::kInf};
    return m;
}

qp::QpModel unboundedModel() {
    qp::QpModel m;
    m.P = qp::SparseMatrix::fromTriplets(2, 2, {}, {}, {});
    m.A = qp::SparseMatrix::fromTriplets(3, 2, {0, 0, 1, 2}, {0, 1, 0, 1}, {1, -1, 1, 1});
    m.q = {-1.0, -1.0};
    m.l = {-qpcontract::kInf, 0.0, 0.0};
    m.u = {1.0, qpcontract::kInf, qpcontract::kInf};
    return m;
}

void solveEquivalence() {
    struct Variant {
        const char* name;
        void (*configure)(qp::AdmmOptions&);
    };
    const Variant variants[] = {
        {"default", [](qp::AdmmOptions&) {}},
        {"noRuiz", [](qp::AdmmOptions& o) { o.useRuizScaling = false; }},
        {"fixedRho", [](qp::AdmmOptions& o) { o.useAdaptiveRho = false; }},
        {"noPolish", [](qp::AdmmOptions& o) { o.usePolishing = false; }},
        {"iterationLimit", [](qp::AdmmOptions& o) { o.iterationLimit = 40; }},
    };
    struct Case {
        std::string name;
        qp::QpModel model;
    };
    std::vector<Case> cases;
    for (qpcontract::Fixture& fixture : qpcontract::fixtures()) {
        cases.push_back({fixture.name, std::move(fixture.model)});
    }
    cases.push_back({"infeasible", infeasibleModel()});
    cases.push_back({"unbounded", unboundedModel()});

    for (const Case& c : cases) {
        for (const Variant& v : variants) {
            run("solve " + c.name + "/" + v.name, [&] {
                qp::AdmmOptions cpuOptions;
                cpuOptions.threadCount = 1;
                cpuOptions.backend = qp::ComputeBackend::Cpu;
                v.configure(cpuOptions);
                qp::AdmmOptions gpuOptions = cpuOptions;
                gpuOptions.backend = qp::ComputeBackend::Cuda;
                const qp::AdmmResult cpu = qp::QpSolver{}.solve(c.model, cpuOptions);
                const qp::AdmmResult gpu = qp::QpSolver{}.solve(c.model, gpuOptions);
                qpcontract::check(gpu.executedBackend == qp::ComputeBackend::Cuda,
                                  "the CUDA solve must report the CUDA backend: " + gpu.backendMessage);
                qpcontract::check(cpu.status == gpu.status,
                                  std::string("status differs: cpu ") + qp::toString(cpu.status) +
                                      ", cuda " + qp::toString(gpu.status));
                if (cpu.status == qp::QpStatus::Optimal) {
                    qpcontract::requireClose(cpu.primalObjective, gpu.primalObjective, 1e-6, "objective");
                }
                std::printf("        iterations cpu %lld cuda %lld; cuda kkt %.3g s, setup %.3g s, H2D %lld B, D2H %lld B\n",
                            static_cast<long long>(cpu.iterations), static_cast<long long>(gpu.iterations),
                            gpu.kktSolveSeconds, gpu.backendProfile.setupSeconds,
                            static_cast<long long>(gpu.backendProfile.hostToDeviceBytes),
                            static_cast<long long>(gpu.backendProfile.deviceToHostBytes));
            });
        }
    }
}

}  // namespace

int main() {
    const qp::CudaAvailability availability = qp::cudaAvailability(0);
    if (!availability.usable) {
        std::printf("SKIPPED: no usable CUDA device (%s)\n", availability.reason.c_str());
        return 77;
    }
    std::printf("CUDA device 0: %s\n", availability.deviceName.c_str());

    backendEquivalence();
    solveEquivalence();

    if (failures == 0) {
        std::printf("All QP CUDA tests passed\n");
        return EXIT_SUCCESS;
    }
    std::printf("%d test(s) failed\n", failures);
    return EXIT_FAILURE;
}
