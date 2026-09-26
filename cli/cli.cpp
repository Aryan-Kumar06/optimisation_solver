#include "cli.h"
#include "argument_parser.h"
#include "mascot.h"
#include "json_report.h"

#include "model/model.h"
#include "mps/mps_reader.h"
#include "solver/classifier.h"
#include "solver/dispatcher.h"
#include "solver/orchestrator.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

namespace cli {
namespace {

std::string trim(const std::string& str) {
    auto start = str.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) return "";
    auto end = str.find_last_not_of(" \t\r\n");
    return str.substr(start, end - start + 1);
}

std::string expandPath(std::string path) {
    path = trim(path);
    if (path.empty()) return path;

    if (path == "~") {
        const char* home = std::getenv("HOME");
        return home ? std::string(home) : path;
    }
    if (path.rfind("~/", 0) == 0) {
        const char* home = std::getenv("HOME");
        if (home) {
            return std::string(home) + path.substr(1);
        }
    }
    return path;
}

bool exportSolutionToFile(const model::Model& model,
                          const solver::SolveResult& solveResult,
                          const std::string& outputPath,
                          std::ostream& out,
                          std::ostream& err,
                          const TerminalStyle& style) {
    std::ofstream outFile(outputPath);
    if (!outFile.is_open()) {
        printError(err, "Output error", "Unable to open output file '" + outputPath + "' for writing.");
        return false;
    }

    outFile << "# Solution for " << (model.name.empty() ? "model" : model.name) << "\n";
    outFile << "# Status: " << solver::toString(solveResult.status) << "\n";
    outFile << "# Objective: " << std::setprecision(9) << solveResult.objectiveValue << "\n";
    for (std::size_t i = 0; i < model.variables.size(); ++i) {
        const std::string& name = model.variables[i].name.empty() ? ("x" + std::to_string(i)) : model.variables[i].name;
        outFile << name << " " << std::setprecision(9) << solveResult.variableValues[i] << "\n";
    }
    if (solveResult.hasDuals) {
        for (std::size_t i = 0; i < model.constraints.size(); ++i) {
            const auto name = model.constraints[i].name.empty()
                ? "c" + std::to_string(i) : model.constraints[i].name;
            outFile << "# Dual " << name << " " << solveResult.constraintDuals[i] << "\n";
        }
        for (std::size_t j = 0; j < model.variables.size(); ++j) {
            const auto name = model.variables[j].name.empty()
                ? "x" + std::to_string(j) : model.variables[j].name;
            outFile << "# Reduced cost " << name << " " << solveResult.reducedCosts[j] << "\n";
        }
    } else {
        outFile << "# Duals unavailable: " << solveResult.dualsUnavailableReason << "\n";
    }
    outFile.close();
    if (!outFile) {
        printError(err, "Output error", "Failed to write complete solution to '" + outputPath + "'.");
        return false;
    }
    printSolutionWritten(out, outputPath, style);
    return true;
}

// Writes the structured record when --json was given. Every exit path below
// goes through this: a refusal, a proof of infeasibility and a timeout are all
// OUTCOMES a benchmark has to record, and returning without one makes them
// indistinguishable from a crash.
bool emitJsonRecord(const JsonReportInput& report,
                    const solver::SolveResult& solveResult,
                    const std::optional<std::string>& jsonPath,
                    std::ostream& err) {
    if (!jsonPath.has_value()) {
        return true;
    }
    std::ofstream jsonFile(*jsonPath);
    if (!jsonFile.is_open()) {
        printError(err, "Output error",
                   "Unable to open JSON file '" + *jsonPath + "' for writing.");
        return false;
    }
    writeJsonReport(jsonFile, report, solveResult);
    jsonFile.close();
    if (!jsonFile) {
        printError(err, "Output error",
                   "Failed to write the JSON record to '" + *jsonPath + "'.");
        return false;
    }
    return true;
}

int solveModel(const model::Model& model,
               const solver::SolverOptions& solverOptions,
               const std::optional<std::string>& outputPath,
               std::ostream& out,
               std::ostream& err,
               const TerminalStyle& style,
               JsonReportInput* report = nullptr,
               const std::optional<std::string>& jsonPath = std::nullopt) {
    const auto emit = [&](const solver::SolveResult& r) {
        return report == nullptr || emitJsonRecord(*report, r, jsonPath, err);
    };

    solver::SolveResult solveResult;
    const auto solveStart = std::chrono::steady_clock::now();
    try {
        solveResult = solver::solve(model, solverOptions);
    } catch (const std::exception& ex) {
        printError(err, "Solver failed", ex.what());
        return 1;
    }
    if (report != nullptr) {
        report->solveSeconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - solveStart).count();
    }

    if (solveResult.status == solver::SolveStatus::InvalidModel) {
        printError(err, "Solver error: Invalid model", solveResult.message);
        emit(solveResult);
        return 1;
    }
    if (solveResult.status == solver::SolveStatus::NumericalFailure) {
        printError(err, "Solver error: Numerical failure", solveResult.message);
        emit(solveResult);
        return 1;
    }
    if (solveResult.status == solver::SolveStatus::Unsupported) {
        printError(err, "Solver error: Unsupported problem", solveResult.message);
        emit(solveResult);
        return 1;
    }

    SolveDashboardInfo dash;
    dash.problemName = model.name;
    dash.originalVars = model.variables.size();
    dash.originalCons = model.constraints.size();
    dash.reducedVars = solveResult.reducedVariableCount;
    dash.reducedCons = solveResult.reducedConstraintCount;
    dash.presolveInfeasible = solveResult.engine == solver::Engine::Infeasible;
    dash.engineName = dash.presolveInfeasible ? "presolve"
                                            : solver::toString(solveResult.executedEngine);
    printSolveDashboard(out, dash, style);

    SolveResultInfo resInfo;
    resInfo.status = solveResult.status;
    resInfo.objective = solveResult.objectiveValue;
    resInfo.hasObjective = solveResult.hasPrimal;
    resInfo.engine = dash.engineName;
    resInfo.iterations = solveResult.iterations;
    resInfo.nodeCount = solveResult.nodeCount;
    resInfo.solveSeconds = solveResult.solveSeconds;
    resInfo.message = solveResult.message;
    printSolveResult(out, resInfo, style);

    if (!solveResult.hasPrimal) {
        if (solveResult.status == solver::SolveStatus::LimitReached) {
            out << "  No feasible solution available.\n";
            if (outputPath.has_value()) {
                printError(err, "Output unavailable", "No feasible solution was found to write.");
                emit(solveResult);
                return 1;
            }
        }
        // Proved infeasible, proved unbounded, or a limit with no incumbent:
        // all are legitimate outcomes and each still has to produce a record.
        return emit(solveResult) ? 0 : 1;
    }

    if (solveResult.hasDuals) {
        out << "  Duals available: " << solveResult.constraintDuals.size()
            << " shadow prices, " << solveResult.reducedCosts.size() << " reduced costs\n";
    } else {
        out << "  Duals unavailable: " << solveResult.dualsUnavailableReason << "\n";
    }
    if (!solveResult.integralityRespected) {
        out << "  Continuous relaxation; returned values are not integer-feasible.\n";
    }

    if (outputPath.has_value()) {
        if (!exportSolutionToFile(model, solveResult, *outputPath, out, err, style)) {
            emit(solveResult);
            return 1;
        }
    }

    return emit(solveResult) ? 0 : 1;
}

int solveFile(const std::string& modelPath,
              const std::optional<std::string>& solverName,
              const std::optional<double>& timeLimitSeconds,
              const std::optional<std::string>& outputPath,
              std::ostream& out,
              std::ostream& err,
              const TerminalStyle& style,
              const std::optional<std::string>& jsonPath,
              const std::optional<std::string>& dumpModelPath,
              const std::optional<int>& threadCount) {
    mps::MpsReader reader;
    model::Model model;
    const auto parseStart = std::chrono::steady_clock::now();
    try {
        model = reader.read(modelPath);
    } catch (const std::exception& ex) {
        printError(err, "Failed to read MPS file", "Path: " + modelPath + "\n" + ex.what());
        return 1;
    }
    const double parseSeconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - parseStart).count();

    if (!model.validate()) {
        printError(err, "Invalid model", "Model failed structural validation.");
        return 1;
    }

    // Parse-only mode: dump what the reader produced and stop. Deliberately
    // before any solving, so the dump reflects the ORIGINAL model untouched by
    // presolve -- it exists to be diffed against an independent reader.
    if (dumpModelPath.has_value()) {
        std::ofstream dumpFile(*dumpModelPath);
        if (!dumpFile.is_open()) {
            printError(err, "Output error",
                       "Unable to open '" + *dumpModelPath + "' for writing.");
            return 1;
        }
        writeModelDump(dumpFile, model);
        dumpFile.close();
        if (!dumpFile) {
            printError(err, "Output error", "Failed to write the model dump.");
            return 1;
        }
        return 0;
    }

    solver::SolverOptions solverOptions;
    if (solverName.has_value()) {
        solverOptions.forceEngine = solver::parseEngine(*solverName);
    }
    if (timeLimitSeconds.has_value()) {
        solverOptions.timeLimitSeconds = *timeLimitSeconds;
    }
    if (threadCount.has_value()) {
        solverOptions.threadCount = *threadCount;
    }

    JsonReportInput report;
    report.instancePath = modelPath;
    report.requestedEngine = solverName.value_or(std::string{});
    report.timeLimitSeconds = timeLimitSeconds.value_or(0.0);
    report.threadCount = threadCount.value_or(0);
    report.tolerance = solverOptions.tolerance;
    report.parseSeconds = parseSeconds;
    report.originalVariables = model.variables.size();
    report.originalConstraints = model.constraints.size();
    report.originalModel = &model;
    if (jsonPath.has_value()) {
        // Only hashed when a record is actually being written; it is a full
        // pass over the file and pointless otherwise.
        report.instanceSha256 = sha256File(modelPath);
    }

    return solveModel(model, solverOptions, outputPath, out, err, style,
                      &report, jsonPath);
}

// Session state maintained throughout interactive mode
struct InteractiveSession {
    bool hasModel = false;
    std::string modelPath;
    std::string modelName;
    model::Model model;
    solver::Classification classification;

    std::optional<std::string> forcedEngine;
    std::optional<double> timeLimitSeconds;
    std::optional<std::string> outputPath;

    bool hasLastResult = false;
    solver::SolveResult lastResult;
};

void displayCurrentModelCard(std::ostream& out, const InteractiveSession& session, const TerminalStyle& s) {
    printHeaderBox(out, "CURRENT MODEL", s);
    std::int64_t nonzeros = 0;
    for (const auto& con : session.model.constraints) {
        nonzeros += con.linearTerms.size();
    }
    std::string fileName = session.modelPath;
    auto slashPos = fileName.find_last_of("/\\");
    if (slashPos != std::string::npos) {
        fileName = fileName.substr(slashPos + 1);
    }

    out << "  " << s.dim() << "File             " << s.reset() << fileName << "\n";
    out << "  " << s.dim() << "Problem type     " << s.reset() << s.bold()
        << solver::toString(session.classification.problemClass) << s.reset() << "\n";
    out << "  " << s.dim() << "Variables        " << s.reset()
        << formatNumber(session.model.variables.size()) << "\n";
    out << "  " << s.dim() << "Constraints      " << s.reset()
        << formatNumber(session.model.constraints.size()) << "\n";
    out << "  " << s.dim() << "Nonzeros         " << s.reset()
        << formatNumber(nonzeros) << "\n\n";
}

void displayModelInfoScreen(std::ostream& out, std::istream& in, const InteractiveSession& session, const TerminalStyle& s) {
    printHeaderBox(out, "MODEL INFORMATION", s);
    if (!session.hasModel) {
        out << "  " << s.boldYellow() << "No model is currently loaded." << s.reset() << "\n";
        out << "  Please select " << s.boldCyan() << "[1] Open MPS Model" << s.reset() << " first.\n\n";
        out << "  " << s.dim() << "Press Enter to return..." << s.reset();
        std::string dummy;
        std::getline(in, dummy);
        out << "\n";
        return;
    }

    std::string fileName = session.modelPath;
    auto slashPos = fileName.find_last_of("/\\");
    if (slashPos != std::string::npos) {
        fileName = fileName.substr(slashPos + 1);
    }

    int nContinuous = 0, nInteger = 0, nBinary = 0;
    for (const auto& var : session.model.variables) {
        if (var.type == model::VariableType::Binary) {
            ++nBinary;
        } else if (var.type == model::VariableType::Integer) {
            ++nInteger;
        } else {
            ++nContinuous;
        }
    }

    std::int64_t nonzeros = 0;
    for (const auto& con : session.model.constraints) {
        nonzeros += con.linearTerms.size();
    }

    out << "  " << s.bold() << "File" << s.reset() << "\n";
    out << "      " << fileName << "\n\n";

    out << "  " << s.bold() << "Problem type" << s.reset() << "\n";
    out << "      " << solver::toString(session.classification.problemClass) << "\n\n";

    out << "  " << s.bold() << "Variables" << s.reset() << "\n";
    out << "      " << formatNumber(session.model.variables.size()) << "\n";
    out << "      Continuous    " << formatNumber(nContinuous) << "\n";
    out << "      Integer       " << formatNumber(nInteger) << "\n";
    out << "      Binary        " << formatNumber(nBinary) << "\n\n";

    out << "  " << s.bold() << "Constraints" << s.reset() << "\n";
    out << "      " << formatNumber(session.model.constraints.size()) << "\n\n";

    out << "  " << s.bold() << "Matrix nonzeros" << s.reset() << "\n";
    out << "      " << formatNumber(nonzeros) << "\n\n";

    out << "  " << s.bold() << "Objective" << s.reset() << "\n";
    out << "      " << (session.model.objective.sense == model::ObjectiveSense::Minimize ? "Minimize" : "Maximize") << "\n";
    out << "      Quadratic: " << (session.model.objective.quadraticTerms.empty() ? "No" : "Yes") << "\n\n";

    out << "  " << s.dim() << "Press Enter to return..." << s.reset();
    std::string dummy;
    std::getline(in, dummy);
    out << "\n";
}

void displaySettingsScreen(std::ostream& out, std::istream& in, InteractiveSession& session, const TerminalStyle& s) {
    while (in.good()) {
        printHeaderBox(out, "SOLVER SETTINGS", s);

        std::string solverStr = session.forcedEngine.has_value() ? *session.forcedEngine : "Automatic";
        std::string timeStr = session.timeLimitSeconds.has_value() ? (std::to_string(*session.timeLimitSeconds) + " seconds") : "None";
        std::string outStr = session.outputPath.has_value() ? *session.outputPath : "Not specified";

        out << "  " << s.bold() << "Solver" << s.reset() << "\n";
        out << "      " << solverStr << "\n\n";
        out << "  " << s.bold() << "Time limit" << s.reset() << "\n";
        out << "      " << timeStr << "\n\n";
        out << "  " << s.bold() << "Output file" << s.reset() << "\n";
        out << "      " << outStr << "\n\n";

        out << "  " << s.boldCyan() << "[1]" << s.reset() << " Change solver\n";
        out << "  " << s.boldCyan() << "[2]" << s.reset() << " Change time limit\n";
        out << "  " << s.boldCyan() << "[3]" << s.reset() << " Change output file\n";
        out << "  " << s.boldCyan() << "[4]" << s.reset() << " Back\n\n";
        out << "  " << s.dim() << "Select an option [1-4]: " << s.reset();

        std::string choice;
        if (!std::getline(in, choice)) break;
        choice = trim(choice);

        if (choice == "1") {
            out << "\n  Available: auto, pdlp, dual_simplex, branch_and_cut, qp\n";
            out << "  Enter solver (or 'auto' for default): ";
            std::string eng;
            if (!std::getline(in, eng)) break;
            eng = trim(eng);
            if (eng == "auto" || eng.empty()) {
                session.forcedEngine.reset();
                out << "  " << s.boldGreen() << "✓" << s.reset() << " Solver set to automatic dispatch.\n\n";
            } else if (ArgumentParser::isValidSolverName(eng)) {
                session.forcedEngine = eng;
                out << "  " << s.boldGreen() << "✓" << s.reset() << " Solver override set to: " << eng << "\n\n";
            } else {
                out << "  " << s.boldRed() << "✗ Invalid solver name." << s.reset() << "\n\n";
            }
        } else if (choice == "2") {
            out << "\n  Enter time limit in seconds (0 for no limit): ";
            std::string tlStr;
            if (!std::getline(in, tlStr)) break;
            try {
                double tl = std::stod(trim(tlStr));
                if (tl <= 0.0) {
                    session.timeLimitSeconds.reset();
                    out << "  " << s.boldGreen() << "✓" << s.reset() << " Time limit removed.\n\n";
                } else {
                    session.timeLimitSeconds = tl;
                    out << "  " << s.boldGreen() << "✓" << s.reset() << " Time limit set to " << tl << " seconds.\n\n";
                }
            } catch (...) {
                out << "  " << s.boldRed() << "✗ Invalid number format." << s.reset() << "\n\n";
            }
        } else if (choice == "3") {
            out << "\n  Enter output file path (or 'none' to clear): ";
            std::string pStr;
            if (!std::getline(in, pStr)) break;
            pStr = trim(pStr);
            if (pStr == "none" || pStr.empty()) {
                session.outputPath.reset();
                out << "  " << s.boldGreen() << "✓" << s.reset() << " Output file cleared.\n\n";
            } else {
                session.outputPath = expandPath(pStr);
                out << "  " << s.boldGreen() << "✓" << s.reset() << " Output file set to: " << *session.outputPath << "\n\n";
            }
        } else if (choice == "4" || choice == "back" || choice == "b") {
            out << "\n";
            break;
        } else {
            out << "  " << s.boldYellow() << "Unrecognised option. Please select 1 to 4." << s.reset() << "\n\n";
        }
    }
}

void displayHelpScreen(std::ostream& out, std::istream& in, const TerminalStyle& s) {
    printHeaderBox(out, "HELP", s);

    out << "  " << s.bold() << "Interactive mode:" << s.reset() << "\n";
    out << "      Run `optimsolver` with no command.\n\n";

    out << "  " << s.bold() << "Command-line mode:" << s.reset() << "\n";
    out << "      optimsolver solve <model.mps>\n\n";

    out << "  " << s.bold() << "Solver override:" << s.reset() << "\n";
    out << "      --solver pdlp\n";
    out << "      --solver dual_simplex\n";
    out << "      --solver branch_and_cut\n";
    out << "      --solver qp\n\n";

    out << "  " << s.bold() << "Other options:" << s.reset() << "\n";
    out << "      --time-limit <seconds>\n";
    out << "      --output <file>\n\n";

    out << "  " << s.bold() << "Documentation:" << s.reset() << "\n";
    out << "      USER_GUIDE.md        Complete usage guide & MPS specification\n";
    out << "      docs/architecture.md System architecture & engine details\n\n";

    out << "  " << s.dim() << "Press Enter to return..." << s.reset();
    std::string dummy;
    std::getline(in, dummy);
    out << "\n";
}

bool openModelPrompt(std::ostream& out, std::istream& in, InteractiveSession& session, const TerminalStyle& s) {
    while (in.good()) {
        printHeaderBox(out, "OPEN MPS MODEL", s);
        out << "  Enter path to an MPS file (or press Enter to cancel):\n  > ";
        std::string rawPath;
        if (!std::getline(in, rawPath)) {
            return false;
        }
        std::string path = expandPath(rawPath);
        if (path.empty()) {
            out << "\n  " << s.dim() << "Operation cancelled.\n\n" << s.reset();
            return false;
        }

        mps::MpsReader reader;
        model::Model loadedModel;
        try {
            loadedModel = reader.read(path);
        } catch (const std::exception& ex) {
            out << "\n  " << s.boldRed() << "✗ Could not load model" << s.reset() << "\n\n";
            out << "  Reason:\n  " << ex.what() << "\n\n";
            out << "  " << s.boldCyan() << "[1]" << s.reset() << " Try another file\n";
            out << "  " << s.boldCyan() << "[2]" << s.reset() << " Return to menu\n\n";
            out << "  " << s.dim() << "Select an option [1-2]: " << s.reset();

            std::string sub;
            if (!std::getline(in, sub)) return false;
            sub = trim(sub);
            if (sub == "1") {
                continue;
            }
            out << "\n";
            return false;
        }

        if (!loadedModel.validate()) {
            out << "\n  " << s.boldRed() << "✗ Could not load model" << s.reset() << "\n\n";
            out << "  Reason:\n  Model failed structural validation.\n\n";
            out << "  " << s.boldCyan() << "[1]" << s.reset() << " Try another file\n";
            out << "  " << s.boldCyan() << "[2]" << s.reset() << " Return to menu\n\n";
            out << "  " << s.dim() << "Select an option [1-2]: " << s.reset();

            std::string sub;
            if (!std::getline(in, sub)) return false;
            sub = trim(sub);
            if (sub == "1") {
                continue;
            }
            out << "\n";
            return false;
        }

        session.hasModel = true;
        session.modelPath = path;
        session.modelName = loadedModel.name.empty() ? path : loadedModel.name;
        session.model = std::move(loadedModel);
        session.classification = solver::classify(session.model);
        session.hasLastResult = false;

        std::string fileName = session.modelPath;
        auto slashPos = fileName.find_last_of("/\\");
        if (slashPos != std::string::npos) {
            fileName = fileName.substr(slashPos + 1);
        }
        out << "\n  " << s.boldGreen() << "✓" << s.reset() << " Successfully loaded "
            << s.bold() << fileName << s.reset() << "\n\n";
        return true;
    }
    return false;
}

void solveInteractive(std::ostream& out, std::ostream& err, std::istream& in, InteractiveSession& session, const TerminalStyle& s) {
    if (!session.hasModel) {
        out << "  " << s.boldYellow() << "No model loaded. Please open a model first.\n\n" << s.reset();
        return;
    }

    printHeaderBox(out, "SOLVING", s);

    std::string fileName = session.modelPath;
    auto slashPos = fileName.find_last_of("/\\");
    if (slashPos != std::string::npos) {
        fileName = fileName.substr(slashPos + 1);
    }

    std::string defaultEngine = "dual_simplex / pdlp";
    if (session.classification.problemClass == solver::ProblemClass::MILP) {
        defaultEngine = "branch_and_cut";
    } else if (session.classification.problemClass == solver::ProblemClass::QP) {
        defaultEngine = "qp";
    }

    std::string engineName = session.forcedEngine.has_value() ? *session.forcedEngine : (defaultEngine + " (Auto)");

    out << "  " << s.dim() << "Model              " << s.reset() << fileName << "\n";
    out << "  " << s.dim() << "Problem            " << s.reset() << solver::toString(session.classification.problemClass) << "\n";
    out << "  " << s.dim() << "Engine             " << s.reset() << engineName << "\n\n";

    solver::SolverOptions solverOptions;
    if (session.forcedEngine.has_value()) {
        solverOptions.forceEngine = solver::parseEngine(*session.forcedEngine);
    }
    if (session.timeLimitSeconds.has_value()) {
        solverOptions.timeLimitSeconds = *session.timeLimitSeconds;
    }

    animateSolveProgress(out, s, "Presolving model...");
    animateSolveProgress(out, s, "Executing numerical solver engine...");

    solver::SolveResult solveResult;
    try {
        solveResult = solver::solve(session.model, solverOptions);
    } catch (const std::exception& ex) {
        out << "  " << s.boldRed() << "✗ Solver failed with exception: " << ex.what() << s.reset() << "\n\n";
        return;
    }

    animateSolveProgress(out, s, "Reconstructing solution & duals...");

    session.hasLastResult = true;
    session.lastResult = solveResult;

    // Mascot reaction to solve status
    MascotState mascotEndState = (solveResult.status == solver::SolveStatus::Optimal)
                                     ? MascotState::Success
                                     : MascotState::Error;
    printMascot(out, s, mascotEndState);
    out << "\n";

    // Print FINAL RESULT SCREEN
    printHeaderBox(out, "SOLVE RESULT", s);

    // Status presentation
    if (solveResult.status == solver::SolveStatus::Optimal) {
        out << "  " << s.boldGreen() << "✓ OPTIMAL" << s.reset() << "\n\n";
    } else if (solveResult.status == solver::SolveStatus::Infeasible) {
        out << "  " << s.boldRed() << "✗ INFEASIBLE" << s.reset() << "\n\n";
    } else if (solveResult.status == solver::SolveStatus::Unbounded) {
        out << "  " << s.boldYellow() << "! UNBOUNDED" << s.reset() << "\n\n";
    } else if (solveResult.status == solver::SolveStatus::LimitReached) {
        out << "  " << s.boldYellow() << "! LIMIT REACHED" << s.reset() << "\n\n";
    } else if (solveResult.status == solver::SolveStatus::NumericalFailure) {
        out << "  " << s.boldRed() << "✗ NUMERICAL FAILURE" << s.reset() << "\n\n";
    } else if (solveResult.status == solver::SolveStatus::InvalidModel) {
        out << "  " << s.boldRed() << "✗ INVALID MODEL" << s.reset() << "\n\n";
    } else if (solveResult.status == solver::SolveStatus::Unsupported) {
        out << "  " << s.boldRed() << "✗ UNSUPPORTED" << s.reset() << "\n\n";
    }

    if (!solveResult.message.empty() && solveResult.status != solver::SolveStatus::Optimal) {
        out << "  " << s.dim() << "Reason: " << s.reset() << solveResult.message << "\n\n";
    }

    // Divider
    auto printDivider = [&](const std::string& sectionTitle) {
        out << "  " << s.bold() << sectionTitle << s.reset() << "\n";
        out << "  " << s.dim();
        for (int i = 0; i < 60; ++i) out << "─";
        out << s.reset() << "\n";
    };

    printDivider("MODEL");
    std::string actualEngine = (solveResult.engine == solver::Engine::Infeasible)
                                   ? "presolve"
                                   : solver::toString(solveResult.executedEngine);
    out << "  " << s.dim() << "Problem type          " << s.reset()
        << solver::toString(session.classification.problemClass) << "\n";
    out << "  " << s.dim() << "Engine                " << s.reset()
        << actualEngine << "\n\n";

    printDivider("MODEL SIZE");
    out << "  " << s.dim() << "Variables             " << s.reset()
        << formatNumber(session.model.variables.size()) << " → "
        << formatNumber(solveResult.reducedVariableCount) << "\n";
    out << "  " << s.dim() << "Constraints           " << s.reset()
        << formatNumber(session.model.constraints.size()) << " → "
        << formatNumber(solveResult.reducedConstraintCount) << "\n\n";

    printDivider("SOLUTION");
    if (solveResult.hasPrimal) {
        out << "  " << s.dim() << "Objective             " << s.reset() << s.bold()
            << std::setprecision(9) << solveResult.objectiveValue << s.reset() << "\n";
        out << "  " << s.dim() << "Primal feasibility    " << s.reset() << s.boldGreen() << "✓" << s.reset() << "\n";
    } else {
        out << "  " << s.dim() << "Objective             " << s.reset() << "Not available\n";
        out << "  " << s.dim() << "Primal feasibility    " << s.reset()
            << (solveResult.status == solver::SolveStatus::Infeasible ? s.boldRed() + "✗ (Infeasible)" + s.reset() : "Not available") << "\n";
    }

    // Integrality
    bool hasInt = false;
    for (const auto& v : session.model.variables) {
        if (v.type == model::VariableType::Integer || v.type == model::VariableType::Binary) {
            hasInt = true;
            break;
        }
    }
    if (hasInt) {
        if (solveResult.integralityRespected) {
            out << "  " << s.dim() << "Integrality           " << s.reset() << s.boldGreen() << "✓" << s.reset() << "\n";
        } else {
            out << "  " << s.dim() << "Integrality           " << s.reset() << s.boldYellow() << "Relaxation only" << s.reset() << "\n";
        }
    } else {
        out << "  " << s.dim() << "Integrality           " << s.reset() << "N/A (continuous)\n";
    }

    // Dual feasibility
    if (solveResult.hasDuals) {
        out << "  " << s.dim() << "Dual feasibility      " << s.reset() << s.boldGreen() << "✓" << s.reset()
            << " (" << solveResult.constraintDuals.size() << " shadow prices, "
            << solveResult.reducedCosts.size() << " reduced costs)\n";
    } else if (session.model.constraints.empty() && !hasInt) {
        out << "  " << s.dim() << "Dual feasibility      " << s.reset() << "Empty (zero constraints)\n";
    } else {
        out << "  " << s.dim() << "Dual feasibility      " << s.reset()
            << "Unavailable (" << solveResult.dualsUnavailableReason << ")\n";
    }
    out << "\n";

    printDivider("PERFORMANCE");
    if (solveResult.iterations > 0) {
        out << "  " << s.dim() << "Iterations            " << s.reset() << formatNumber(solveResult.iterations) << "\n";
    }
    if (solveResult.nodeCount > 0) {
        out << "  " << s.dim() << "Nodes                 " << s.reset() << formatNumber(solveResult.nodeCount) << "\n";
    }
    out << "  " << s.dim() << "Solve time            " << s.reset()
        << std::fixed << std::setprecision(4) << solveResult.solveSeconds << " s\n\n";

    printDivider("OUTPUT");
    bool savedToConfigured = false;
    if (session.outputPath.has_value() && solveResult.hasPrimal) {
        if (exportSolutionToFile(session.model, solveResult, *session.outputPath, out, err, s)) {
            savedToConfigured = true;
        }
    }
    if (!savedToConfigured) {
        out << "  " << s.dim() << "Solution file         " << s.reset() << "Not saved\n\n";
        if (solveResult.hasPrimal) {
            out << "  Export solution to file? (enter path or press Enter to continue):\n  > ";
            std::string expPath;
            if (std::getline(in, expPath)) {
                expPath = expandPath(expPath);
                if (!expPath.empty()) {
                    exportSolutionToFile(session.model, solveResult, expPath, out, err, s);
                }
            }
        } else {
            out << "  " << s.dim() << "Press Enter to continue..." << s.reset();
            std::string dummy;
            std::getline(in, dummy);
        }
    } else {
        out << "\n  " << s.dim() << "Press Enter to continue..." << s.reset();
        std::string dummy;
        std::getline(in, dummy);
    }
    out << "\n";
}

int runInteractive(std::ostream& out, std::ostream& err, std::istream& in) {
    TerminalStyle style = TerminalStyle::forStream(out);
    InteractiveSession session;

    while (in.good()) {
        printMascot(out, style, MascotState::Idle);
        out << "\n";
        printHomeScreenBanner(out, style);
        out << "\n";

        if (session.hasModel) {
            displayCurrentModelCard(out, session, style);
        }

        printInteractiveMenu(out, style, session.hasModel);

        std::string choice;
        if (!std::getline(in, choice)) {
            break;
        }
        choice = trim(choice);
        if (choice.empty()) {
            continue;
        }

        if (session.hasModel) {
            if (choice == "1" || choice == "solve") {
                solveInteractive(out, err, in, session, style);
            } else if (choice == "2" || choice == "open") {
                openModelPrompt(out, in, session, style);
            } else if (choice == "3" || choice == "info") {
                displayModelInfoScreen(out, in, session, style);
            } else if (choice == "4" || choice == "settings") {
                displaySettingsScreen(out, in, session, style);
            } else if (choice == "5" || choice == "help") {
                displayHelpScreen(out, in, style);
            } else if (choice == "6" || choice == "exit" || choice == "quit" || choice == "q") {
                out << "Exiting Optimisation Solver.\n";
                break;
            } else {
                out << style.boldYellow() << "Unrecognised option. Please select 1 to 6." << style.reset() << "\n\n";
            }
        } else {
            if (choice == "1" || choice == "open") {
                openModelPrompt(out, in, session, style);
            } else if (choice == "2" || choice == "settings") {
                displaySettingsScreen(out, in, session, style);
            } else if (choice == "3" || choice == "info") {
                displayModelInfoScreen(out, in, session, style);
            } else if (choice == "4" || choice == "help") {
                displayHelpScreen(out, in, style);
            } else if (choice == "5" || choice == "exit" || choice == "quit" || choice == "q") {
                out << "Exiting Optimisation Solver.\n";
                break;
            } else {
                out << style.boldYellow() << "Unrecognised option. Please select 1 to 5." << style.reset() << "\n\n";
            }
        }
    }

    return 0;
}

}  // namespace

int run(int argc, char* argv[], std::ostream& out, std::ostream& err, std::istream& in) {
    ParseResult parseResult = ArgumentParser::parse(argc, argv);

    if (!parseResult.success) {
        printError(err, parseResult.errorTitle, parseResult.errorDetails);
        return 1;
    }

    if (parseResult.isHelp) {
        if (parseResult.command == Command::Solve) {
            printSolveHelp(out);
        } else {
            printWelcome(out);
        }
        return 0;
    }

    if (parseResult.command == Command::Interactive) {
        return runInteractive(out, err, in);
    }

    if (parseResult.command == Command::Solve) {
        TerminalStyle style = TerminalStyle::forStream(out);
        const auto& opts = parseResult.solveOptions;
        return solveFile(opts.modelPath, opts.solver, opts.timeLimitSeconds, opts.outputPath,
                         out, err, style, opts.jsonPath, opts.dumpModelPath, opts.threadCount);
    }

    printError(err, "Unhandled command", "");
    return 1;
}

}  // namespace cli
