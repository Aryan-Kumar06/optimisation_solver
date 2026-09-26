#pragma once

// Isolated execution of one solve, with a wall-clock watchdog, process-TREE
// termination and peak-memory measurement.
//
// This is benchmark_model/benchmarkcode.cpp promoted into a real library --
// that file was referenced by no CMake target, so none of it was ever built.
// The fork/exec/poll/drain structure is kept; three things are added, each
// because the benchmark contract needs it and the original could not provide
// it:
//
//   PROCESS TREE. The original sent SIGKILL to the child pid alone. A solver
//   that forks workers (ours runs a thread pool, but a reference solver driven
//   through a wrapper script does fork) leaves those children alive, still
//   holding CPU, while the harness records a clean timeout. The child is now
//   given its own process group with setsid() and the whole group is signalled,
//   so nothing outlives its own timeout.
//
//   ESCALATION. SIGKILL cannot be caught, so a solver killed by it never
//   flushes the partial output that explains what it was doing. The watchdog
//   now sends SIGTERM first, waits a grace period, and only then SIGKILLs.
//
//   PEAK MEMORY. Taken from wait4()'s rusage rather than by sampling, so it
//   cannot miss a spike between samples.
//
// POSIX only. The original carried a Windows branch, but tree termination
// there needs a job object and untested process-control code is worse than an
// honest gap, so the harness does not build on Windows. The solver itself is
// unaffected and stays portable.

#include <chrono>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace benchmark {

struct RunLimits {
    // Hard wall-clock budget for the whole process. Zero means no watchdog.
    std::chrono::milliseconds wallClock{0};

    // How long the process is given to exit after SIGTERM before SIGKILL.
    std::chrono::milliseconds gracePeriod{2000};

    // Cap on captured output per stream. A solver stuck in a logging loop can
    // otherwise fill memory faster than the watchdog fires.
    std::size_t maxCapturedBytes = 8u * 1024u * 1024u;
};

// Field names match what benchmark::buildBenchmarkResult already expects, so
// the existing BenchmarkResult conversion keeps working unchanged.
struct ProcessResult {
    int exitCode = -1;

    std::string stdoutOutput;
    std::string stderrOutput;

    std::chrono::milliseconds runtime{0};

    bool timedOut = false;
    bool launchFailed = false;

    // Null when the platform did not report it.
    std::optional<std::int64_t> peakMemoryBytes;

    // Set when the process died from a signal rather than exiting.
    std::optional<int> terminatingSignal;

    // True when the watchdog had to signal the group.
    bool killedProcessTree = false;

    // True when SIGTERM was not enough and SIGKILL followed.
    bool escalatedToKill = false;

    // True when output hit maxCapturedBytes and was cut short.
    bool stdoutTruncated = false;
    bool stderrTruncated = false;

    // Populated when launchFailed, from the child's pre-exec errno.
    std::string launchError;
};

class ProcessManager {
public:
    [[nodiscard]] ProcessResult run(
        const std::string& executable,
        const std::vector<std::string>& arguments,
        const RunLimits& limits
    );
};

}  // namespace benchmark
