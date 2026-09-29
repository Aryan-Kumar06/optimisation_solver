#include "mps/mps_reader.h"
#include "solver/classifier.h"
#include "solver/orchestrator.h"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

int main() {
    try {
        mps::MpsReader reader;
        const model::Model model = reader.read("tests/mps/test_cases/20_miqp.mps");
        if (solver::classify(model).problemClass != solver::ProblemClass::MIQP)
            throw std::runtime_error("MPS model did not classify as MIQP");

        const solver::SolveResult r = solver::solve(model);
        if (r.engine != solver::Engine::Miqp || r.executedEngine != solver::Engine::Miqp)
            throw std::runtime_error("MPS MIQP did not route through MIQP engine");
        if (r.status != solver::SolveStatus::Optimal)
            throw std::runtime_error(std::string("MPS MIQP status: ") + solver::toString(r.status));
        if (!r.hasPrimal || !r.integralityRespected)
            throw std::runtime_error("MPS MIQP returned no valid integer primal");
        // x^2+y^2-3x-y has integer optimum -2 at (1,0) or (2,1).
        if (std::abs(r.objectiveValue + 2.0) > 3e-4)
            throw std::runtime_error("MPS MIQP objective mismatch");
        if (r.variableValues.size() != 2 ||
            std::abs(r.variableValues[0] - std::round(r.variableValues[0])) > 1e-7)
            throw std::runtime_error("MPS integer variable is fractional");

        std::cout << "MPS -> presolve -> MIQP -> postsolve passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& e) {
        std::cerr << "FAIL: " << e.what() << '\n';
        return EXIT_FAILURE;
    }
}
