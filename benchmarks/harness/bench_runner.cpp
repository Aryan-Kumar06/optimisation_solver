// bench_runner -- runs ONE solve in an isolated process and records what
// happened at the process level.
//
// Every solver in the comparison goes through this same binary, ours and the
// reference alike. That is the point: wall clock, peak memory and termination
// are then measured by one mechanism rather than by each adapter reporting its
// own numbers in its own way, which is how benchmark comparisons quietly stop
// being comparisons.
//
// The runner deliberately knows nothing about optimisation. It does not parse
// solver output, decide whether an answer is right, or compare objectives. It
// runs a command, watches the clock, and writes down what it saw. Verification
// is a separate stage with its own independent model of the problem.
//
// Usage:
//   bench_runner --solver <name> --instance <path> --timeout <seconds>
//                --out <record.json> [--solve-json <path written by the child>]
//                -- <command> [args...]

#include "process_manager.h"
#include "benchmark/benchmarkresult.h"

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

void writeJsonString(std::ostream& out, const std::string& value) {
    out << '"';
    for (const char character : value) {
        switch (character) {
            case '"':  out << "\\\""; break;
            case '\\': out << "\\\\"; break;
            case '\n': out << "\\n"; break;
            case '\r': out << "\\r"; break;
            case '\t': out << "\\t"; break;
            default:
                if (static_cast<unsigned char>(character) < 0x20) {
                    out << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(static_cast<unsigned char>(character))
                        << std::dec << std::setfill(' ');
                } else {
                    out << character;
                }
        }
    }
    out << '"';
}

std::string slurp(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        return {};
    }
    std::ostringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

void usage() {
    std::cerr << "usage: bench_runner --solver <name> --instance <path> "
                 "--timeout <seconds> --out <record.json> "
                 "[--solve-json <path>] [--grace <seconds>] -- <command> [args...]\n";
}

}  // namespace

int main(int argc, char** argv) {
    std::string solverName;
    std::string instancePath;
    std::string outPath;
    std::string solveJsonPath;
    double timeoutSeconds = 0.0;
    double graceSeconds = 2.0;

    std::vector<std::string> command;
    bool afterSeparator = false;

    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        if (afterSeparator) {
            command.push_back(argument);
            continue;
        }
        if (argument == "--") {
            afterSeparator = true;
        } else if (argument == "--solver" && i + 1 < argc) {
            solverName = argv[++i];
        } else if (argument == "--instance" && i + 1 < argc) {
            instancePath = argv[++i];
        } else if (argument == "--out" && i + 1 < argc) {
            outPath = argv[++i];
        } else if (argument == "--solve-json" && i + 1 < argc) {
            solveJsonPath = argv[++i];
        } else if (argument == "--timeout" && i + 1 < argc) {
            timeoutSeconds = std::atof(argv[++i]);
        } else if (argument == "--grace" && i + 1 < argc) {
            graceSeconds = std::atof(argv[++i]);
        } else {
            std::cerr << "bench_runner: unknown argument '" << argument << "'\n";
            usage();
            return 2;
        }
    }

    if (command.empty() || outPath.empty() || solverName.empty()) {
        usage();
        return 2;
    }

    benchmark::RunLimits limits;
    limits.wallClock = std::chrono::milliseconds(
        static_cast<long long>(timeoutSeconds * 1000.0));
    limits.gracePeriod = std::chrono::milliseconds(
        static_cast<long long>(graceSeconds * 1000.0));

    const std::vector<std::string> arguments(command.begin() + 1, command.end());

    benchmark::ProcessManager manager;
    const benchmark::ProcessResult processResult =
        manager.run(command.front(), arguments, limits);

    // Reuse the existing conversion rather than re-deriving the status rules.
    const benchmark::BenchmarkResult summary = benchmark::buildBenchmarkResult(
        solverName, instancePath, processResult);

    std::ofstream out(outPath);
    if (!out.is_open()) {
        std::cerr << "bench_runner: cannot open '" << outPath << "' for writing\n";
        return 2;
    }

    out << "{\n";
    out << "  \"schema\": \"optimsolver.run.v1\",\n";
    out << "  \"solver\": "; writeJsonString(out, solverName); out << ",\n";
    out << "  \"instance\": "; writeJsonString(out, instancePath); out << ",\n";

    out << "  \"command\": [";
    for (std::size_t i = 0; i < command.size(); ++i) {
        if (i != 0) { out << ", "; }
        writeJsonString(out, command[i]);
    }
    out << "],\n";

    out << "  \"process\": {\n";
    out << "    \"run_status\": ";
    writeJsonString(out, benchmark::toString(summary.status));
    out << ",\n";
    out << "    \"exit_code\": ";
    if (processResult.launchFailed) { out << "null"; } else { out << processResult.exitCode; }
    out << ",\n";
    out << "    \"terminating_signal\": ";
    if (processResult.terminatingSignal.has_value()) {
        out << *processResult.terminatingSignal;
    } else {
        out << "null";
    }
    out << ",\n";
    out << "    \"timed_out\": " << (processResult.timedOut ? "true" : "false") << ",\n";
    out << "    \"killed_process_tree\": "
        << (processResult.killedProcessTree ? "true" : "false") << ",\n";
    out << "    \"escalated_to_sigkill\": "
        << (processResult.escalatedToKill ? "true" : "false") << ",\n";
    out << "    \"launch_failed\": "
        << (processResult.launchFailed ? "true" : "false") << ",\n";
    out << "    \"launch_error\": ";
    if (processResult.launchError.empty()) {
        out << "null";
    } else {
        writeJsonString(out, processResult.launchError);
    }
    out << ",\n";
    out << "    \"wall_seconds\": "
        << std::setprecision(9)
        << (static_cast<double>(processResult.runtime.count()) / 1000.0) << ",\n";
    out << "    \"timeout_seconds\": ";
    if (timeoutSeconds > 0.0) { out << timeoutSeconds; } else { out << "null"; }
    out << ",\n";
    out << "    \"peak_memory_bytes\": ";
    if (processResult.peakMemoryBytes.has_value()) {
        out << *processResult.peakMemoryBytes;
    } else {
        out << "null";
    }
    out << ",\n";
    out << "    \"stdout_truncated\": "
        << (processResult.stdoutTruncated ? "true" : "false") << ",\n";
    out << "    \"stderr_truncated\": "
        << (processResult.stderrTruncated ? "true" : "false") << "\n";
    out << "  },\n";

    out << "  \"logs\": {\"stdout\": ";
    writeJsonString(out, processResult.stdoutOutput);
    out << ", \"stderr\": ";
    writeJsonString(out, processResult.stderrOutput);
    out << "},\n";

    // The child's own structured record, embedded verbatim when it produced
    // one. Null when it crashed, timed out, or never got that far -- which is
    // exactly the signal a checker needs to refuse to score the row.
    out << "  \"solve\": ";
    const std::string solveJson =
        solveJsonPath.empty() ? std::string{} : slurp(solveJsonPath);
    if (solveJson.empty()) {
        out << "null";
    } else {
        out << solveJson;
    }
    out << "\n}\n";

    out.close();
    if (!out) {
        std::cerr << "bench_runner: failed writing '" << outPath << "'\n";
        return 2;
    }

    std::cout << benchmark::summarize(summary) << "\n";
    return 0;
}
