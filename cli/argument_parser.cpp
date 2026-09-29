#include "argument_parser.h"
#include "solver/dispatcher.h"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <sstream>
#include <limits>
#include <cerrno>

namespace cli {

bool ArgumentParser::isValidSolverName(const std::string& name) {
    return solver::parseEngine(name).has_value();
}

std::string ArgumentParser::getRootHelp() {
    return "Usage: optimsolver [command] [options]\n\n"
           "Commands:\n"
           "  (none)              Launch interactive terminal interface\n"
           "  solve <model.mps>   Solve an MPS or .nlp problem\n\n"
           "  solve-nlp <model.nlp> Solve a smooth nonlinear problem (solve-nlp --help)\n\n"
           "Options:\n"
           "  -h, --help          Show this help message\n\n"
           "Run 'optimsolver solve --help' for options specific to the solve command.";
}

std::string ArgumentParser::getSolveHelp() {
    return "Usage: optimsolver solve <model.mps> [options]\n\n"
           "Arguments:\n"
           "  <model.mps>             Path to input problem file in MPS format (required)\n\n"
           "Options:\n"
           "  --solver <name>         Force a specific solver engine:\n"
           "                          pdlp, dual_simplex, branch_and_cut, qp, nlp\n"
           "  --time-limit <seconds>  Maximum solve time budget in seconds (positive number)\n"
           "  --output <file>         Write reconstructed original-space solution to file\n"
           "  --json <file>           Write a structured JSON record of the solve\n"
           "  --dump-model <file>     Write the parsed model as JSON and exit\n"
           "  --threads <n>           Worker threads (0 = auto, 1 = serial)\n"
           "  .nlp input             Routes to NLP; see solve-nlp --help for its options\n"
           "                         NLP success means first-order stationarity.\n"
           "  -h, --help              Show this help message";
}

bool ArgumentParser::isNlpPath(const std::string& path) {
    const auto dot = path.find_last_of('.');
    if (dot == std::string::npos) return false;
    std::string extension = path.substr(dot);
    for (char& c : extension) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return extension == ".nlp";
}

std::string ArgumentParser::getNlpHelp() {
    return "Usage: optimsolver solve <model.nlp> [options]\n"
           "       optimsolver solve-nlp <file> [options]\n"
           "  --solver nlp           Use elastic SQP (automatic for .nlp files)\n"
           "  --tolerance value      Original-unit KKT tolerance (positive)\n"
           "  --iterations n         NLP iteration limit (nonnegative)\n"
           "  --time-limit seconds   Positive solve time budget\n"
           "  --json file            Write JSON diagnostics\n"
           "  --output file          Write the evaluated iterate and status\n"
           "  -h, --help             Show this help\n"
           "Returns first-order stationarity, not global optimality.\n"
           "--threads and --dump-model are not supported for NLP.\n";
}

ParseResult ArgumentParser::parse(int argc, const char* const argv[]) {
    ParseResult res;

    if (argc <= 1) {
        res.command = Command::Interactive;
        res.success = true;
        return res;
    }

    std::string firstArg = argv[1];

    if (firstArg == "--help" || firstArg == "-h") {
        res.command = Command::Help;
        res.isHelp = true;
        res.success = true;
        return res;
    }

    if (firstArg == "interactive") {
        res.command = Command::Interactive;
        res.success = true;
        return res;
    }

    if (firstArg != "solve" && firstArg != "solve-nlp") {
        res.success = false;
        res.errorTitle = "Unknown command '" + firstArg + "'";
        res.errorDetails = "Run 'optimsolver --help' to see available commands.";
        res.errorMessage = "Error: Unknown command '" + firstArg + "'.\n" + getRootHelp();
        return res;
    }

    res.command = firstArg == "solve-nlp" ? Command::SolveNlp : Command::Solve;

    int i = 2;
    while (i < argc) {
        std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            if (res.command == Command::Solve && isNlpPath(res.solveOptions.modelPath))
                res.command = Command::SolveNlp;
            res.isHelp = true;
            res.solveOptions.help = true;
            res.success = true;
            return res;
        } else if (arg == "--tolerance" || arg == "--iterations") {
            if (i + 1 >= argc) {
                res.errorTitle = "Missing value for option '" + arg + "'";
                res.errorMessage = res.errorTitle;
                return res;
            }
            const std::string value = argv[++i];
            char* end = nullptr;
            errno = 0;
            if (arg == "--tolerance") {
                double parsed = std::strtod(value.c_str(), &end);
                if (end == value.c_str() || *end || errno == ERANGE || !std::isfinite(parsed) || parsed <= 0) {
                    res.errorTitle = "Invalid NLP tolerance '" + value + "'";
                    res.errorMessage = res.errorTitle;
                    return res;
                }
                res.solveOptions.tolerance = parsed;
            } else {
                long parsed = std::strtol(value.c_str(), &end, 10);
                if (end == value.c_str() || *end || errno == ERANGE || parsed < 0 || parsed > std::numeric_limits<int>::max()) {
                    res.errorTitle = "Invalid NLP iteration limit '" + value + "'";
                    res.errorMessage = res.errorTitle;
                    return res;
                }
                res.solveOptions.iterationLimit = static_cast<int>(parsed);
            }
        } else if (arg == "--solver") {
            if (i + 1 >= argc) {
                res.success = false;
                res.errorTitle = "Missing value for option '--solver'";
                res.errorDetails = "Supported engines: pdlp, dual_simplex, branch_and_cut, qp, nlp";
                res.errorMessage = "Error: Missing value for option '--solver'.";
                return res;
            }
            std::string solverVal = argv[++i];
            if (solverVal.empty() || solverVal[0] == '-') {
                res.success = false;
                res.errorTitle = "Missing value for option '--solver'";
                res.errorDetails = "Supported engines: pdlp, dual_simplex, branch_and_cut, qp, nlp";
                res.errorMessage = "Error: Missing value for option '--solver'.";
                return res;
            }
            if (!isValidSolverName(solverVal)) {
                res.success = false;
                res.errorTitle = "Invalid solver '" + solverVal + "'";
                res.errorDetails = "Supported engines: pdlp, dual_simplex, branch_and_cut, qp, nlp";
                res.errorMessage = "Error: Invalid solver '" + solverVal +
                                   "'. Supported solvers: pdlp, dual_simplex, branch_and_cut, qp, nlp.";
                return res;
            }
            res.solveOptions.solver = solverVal;
        } else if (arg == "--time-limit") {
            if (i + 1 >= argc) {
                res.success = false;
                res.errorTitle = "Missing value for option '--time-limit'";
                res.errorDetails = "Time limit must be a positive finite number of seconds.";
                res.errorMessage = "Error: Missing value for option '--time-limit'.";
                return res;
            }
            std::string limitStr = argv[++i];
            char* endPtr = nullptr;
            double limitVal = std::strtod(limitStr.c_str(), &endPtr);
            if (endPtr == limitStr.c_str() || *endPtr != '\0' || !std::isfinite(limitVal) || limitVal <= 0.0) {
                res.success = false;
                res.errorTitle = "Invalid time limit '" + limitStr + "'";
                res.errorDetails = "Must be a positive finite number of seconds.";
                res.errorMessage = "Error: Invalid time limit '" + limitStr +
                                   "'. Must be a positive number.";
                return res;
            }
            res.solveOptions.timeLimitSeconds = limitVal;
        } else if (arg == "--json" || arg == "--dump-model") {
            const std::string flag = arg;
            if (i + 1 >= argc || std::string(argv[i + 1]).empty() || argv[i + 1][0] == '-') {
                res.success = false;
                res.errorTitle = "Missing value for option '" + flag + "'";
                res.errorDetails = "A file path is required.";
                res.errorMessage = "Error: Missing value for option '" + flag + "'.";
                return res;
            }
            if (flag == "--json") {
                res.solveOptions.jsonPath = argv[++i];
            } else {
                res.solveOptions.dumpModelPath = argv[++i];
            }
        } else if (arg == "--threads") {
            if (i + 1 >= argc) {
                res.success = false;
                res.errorTitle = "Missing value for option '--threads'";
                res.errorDetails = "Thread count must be a non-negative integer (0 = auto).";
                res.errorMessage = "Error: Missing value for option '--threads'.";
                return res;
            }
            const std::string threadsVal = argv[++i];
            char* threadsEnd = nullptr;
            const long threads = std::strtol(threadsVal.c_str(), &threadsEnd, 10);
            if (threadsEnd == threadsVal.c_str() || *threadsEnd != '\0' ||
                threads < 0 || threads > 4096) {
                res.success = false;
                res.errorTitle = "Invalid thread count '" + threadsVal + "'";
                res.errorDetails = "Must be a non-negative integer (0 = auto).";
                res.errorMessage = "Error: Invalid thread count '" + threadsVal + "'.";
                return res;
            }
            res.solveOptions.threadCount = static_cast<int>(threads);
        } else if (arg == "--output") {
            if (i + 1 >= argc) {
                res.success = false;
                res.errorTitle = "Missing value for option '--output'";
                res.errorDetails = "A file path is required to write the solution.";
                res.errorMessage = "Error: Missing value for option '--output'.";
                return res;
            }
            std::string outVal = argv[++i];
            if (outVal.empty() || outVal[0] == '-') {
                res.success = false;
                res.errorTitle = "Missing value for option '--output'";
                res.errorDetails = "A file path is required to write the solution.";
                res.errorMessage = "Error: Missing value for option '--output'.";
                return res;
            }
            res.solveOptions.outputPath = outVal;
        } else if (!arg.empty() && arg[0] == '-') {
            res.success = false;
            res.errorTitle = "Unknown option '" + arg + "'";
            res.errorDetails = "Run 'optimsolver solve --help' to see available options.";
            res.errorMessage = "Error: Unknown option '" + arg + "'.\n" + getSolveHelp();
            return res;
        } else {
            // Positional argument: model path
            if (res.solveOptions.modelPath.empty()) {
                res.solveOptions.modelPath = arg;
            } else {
                res.success = false;
                res.errorTitle = "Unexpected argument '" + arg + "'";
                res.errorDetails = "Exactly one model path is required.";
                res.errorMessage = "Error: Unexpected argument '" + arg +
                                   "'. Exactly one model path is required.";
                return res;
            }
        }
        ++i;
    }

    if (res.solveOptions.modelPath.empty()) {
        res.success = false;
        res.errorTitle = "Missing required model path";
        res.errorDetails = "Usage: optimsolver solve <model.mps> [options]";
        res.errorMessage = "Error: Missing required model path for 'solve' command.\n" +
                           getSolveHelp();
        return res;
    }

    if (res.command == Command::Solve && isNlpPath(res.solveOptions.modelPath))
        res.command = Command::SolveNlp;
    if (res.command == Command::Solve && (res.solveOptions.tolerance || res.solveOptions.iterationLimit)) {
        res.errorTitle = "--tolerance and --iterations currently require a nonlinear model";
        res.errorMessage = res.errorTitle;
        return res;
    }
    res.success = true;
    return res;
}

}  // namespace cli
