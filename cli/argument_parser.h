#pragma once

#include <optional>
#include <string>
#include <vector>

namespace cli {

enum class Command {
    None,
    Help,
    Solve,
    Interactive
};

struct SolveOptions {
    std::string modelPath;
    std::optional<std::string> solver;
    std::optional<double> timeLimitSeconds;
    std::optional<std::string> outputPath;

    // Machine-readable outputs for the benchmark harness. Separate from
    // --output, which stays a human-readable solution listing.
    std::optional<std::string> jsonPath;
    std::optional<std::string> dumpModelPath;
    std::optional<int> threadCount;

    bool help = false;
};

struct ParseResult {
    Command command = Command::None;
    SolveOptions solveOptions;
    bool success = false;
    bool isHelp = false;
    std::string errorMessage;
    std::string errorTitle;
    std::string errorDetails;
};

class ArgumentParser {
public:
    static ParseResult parse(int argc, const char* const argv[]);
    static std::string getRootHelp();
    static std::string getSolveHelp();
    static bool isValidSolverName(const std::string& name);

private:
};

}  // namespace cli
