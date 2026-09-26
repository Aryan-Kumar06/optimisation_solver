#pragma once

#include "solver/orchestrator.h"

#include <iostream>
#include <string>

namespace cli {

// Terminal styling supporting subtle ANSI colors with TTY and NO_COLOR detection.
struct TerminalStyle {
    bool color = false;

    static TerminalStyle forStream(const std::ostream& stream);

    std::string reset() const;
    std::string bold() const;
    std::string dim() const;
    std::string cyan() const;
    std::string boldCyan() const;
    std::string green() const;
    std::string boldGreen() const;
    std::string yellow() const;
    std::string boldYellow() const;
    std::string red() const;
    std::string boldRed() const;
};

// Mascot states for animation and status display
enum class MascotState {
    Idle,
    Loading,
    Presolving,
    Solving,
    Success,
    Error
};

// Mascot and banner rendering
void printMascot(std::ostream& out, const TerminalStyle& style, MascotState state = MascotState::Idle);
void printBanner(std::ostream& out);

// Box drawing and headers
void printHeaderBox(std::ostream& out, const std::string& title, const TerminalStyle& style, int width = 64);
void printHomeScreenBanner(std::ostream& out, const TerminalStyle& style, int width = 64);

// Animated stage progression (only active when output is interactive TTY and not NO_COLOR)
void animateSolveProgress(std::ostream& out, const TerminalStyle& style, const std::string& stage);

// Number formatting helper with commas (e.g. 1248 -> "1,248")
std::string formatNumber(std::int64_t n);

// High-level CLI screen renderers
void printWelcome(std::ostream& out);
void printSolveHelp(std::ostream& out);
void printInteractiveMenu(std::ostream& out, const TerminalStyle& style, bool hasModel = false);

// Solve dashboard and result presentation
struct SolveDashboardInfo {
    std::string problemName;
    std::size_t originalVars = 0;
    std::size_t originalCons = 0;
    std::size_t reducedVars = 0;
    std::size_t reducedCons = 0;
    bool presolveInfeasible = false;
    std::string engineName;
};

void printSolveDashboard(std::ostream& out, const SolveDashboardInfo& info, const TerminalStyle& style);

struct SolveResultInfo {
    solver::SolveStatus status = solver::SolveStatus::Optimal;
    double objective = 0.0;
    bool hasObjective = false;
    int iterations = 0;
    int nodeCount = 0;
    double solveSeconds = 0.0;
    std::string engine;
    std::string message;
};

void printSolveResult(std::ostream& out, const SolveResultInfo& result, const TerminalStyle& style);
void printSolutionWritten(std::ostream& out, const std::string& path, const TerminalStyle& style);

// Professional error output
void printError(std::ostream& err, const std::string& title, const std::string& details = "");

}  // namespace cli
