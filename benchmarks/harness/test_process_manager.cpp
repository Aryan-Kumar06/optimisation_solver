// Tests for the isolation layer.
//
// These assert the three properties the benchmark contract actually depends on
// and that the original benchmark_model code did not provide: a watchdog that
// fires, termination that reaches the whole process tree, and a peak-memory
// number that is real rather than a placeholder.

#include "process_manager.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& what) {
    if (condition) {
        std::cout << "[PASS] " << what << "\n";
    } else {
        std::cout << "[FAIL] " << what << "\n";
        ++failures;
    }
}

benchmark::ProcessResult runShell(const std::string& script,
                                  double timeoutSeconds,
                                  double graceSeconds = 0.5) {
    benchmark::RunLimits limits;
    limits.wallClock = std::chrono::milliseconds(
        static_cast<long long>(timeoutSeconds * 1000.0));
    limits.gracePeriod = std::chrono::milliseconds(
        static_cast<long long>(graceSeconds * 1000.0));
    benchmark::ProcessManager manager;
    return manager.run("/bin/sh", {"-c", script}, limits);
}

}  // namespace

int main() {
    // ---- clean exit -------------------------------------------------------
    {
        const auto result = runShell("printf 'hello'; exit 0", 10.0);
        check(!result.launchFailed, "clean run: launch succeeded");
        check(!result.timedOut, "clean run: did not time out");
        check(result.exitCode == 0, "clean run: exit code 0");
        check(result.stdoutOutput == "hello", "clean run: stdout captured");
    }

    // ---- non-zero exit and stderr ----------------------------------------
    {
        const auto result = runShell("printf 'bad' >&2; exit 7", 10.0);
        check(result.exitCode == 7, "failure run: exit code preserved");
        check(result.stderrOutput == "bad", "failure run: stderr captured");
    }

    // ---- launch failure ---------------------------------------------------
    {
        benchmark::RunLimits limits;
        limits.wallClock = std::chrono::milliseconds(5000);
        benchmark::ProcessManager manager;
        const auto result =
            manager.run("/nonexistent/definitely-not-a-binary", {}, limits);
        check(result.launchFailed, "missing binary: reported as launch failure");
        check(!result.launchError.empty(), "missing binary: errno text recorded");
        check(!result.timedOut, "missing binary: not misreported as a timeout");
    }

    // ---- watchdog fires ---------------------------------------------------
    {
        const auto start = std::chrono::steady_clock::now();
        const auto result = runShell("sleep 30", 1.0);
        const double seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - start).count();
        check(result.timedOut, "watchdog: timeout reported");
        check(result.killedProcessTree, "watchdog: process group signalled");
        check(seconds < 5.0, "watchdog: returned promptly, not after 30s");
    }

    // ---- SIGTERM first, then SIGKILL --------------------------------------
    {
        // Traps SIGTERM and ignores it, so only escalation can end it.
        const auto result = runShell("trap '' TERM; sleep 30", 0.5, 0.5);
        check(result.timedOut, "escalation: timeout reported");
        check(result.escalatedToKill, "escalation: SIGKILL followed SIGTERM");
    }

    // ---- PROCESS TREE termination ----------------------------------------
    //
    // The property the original code lacked. A grandchild that outlives its
    // parent keeps touching a file; after the watchdog fires, that file must
    // stop changing. Killing only the direct child would leave it running.
    {
        const std::string marker = "/tmp/optimsolver_tree_marker.txt";
        std::remove(marker.c_str());

        // Parent exits immediately; the grandchild keeps writing for 30s.
        const std::string script =
            "( for i in $(seq 1 300); do echo $i > " + marker +
            "; sleep 0.1; done ) & sleep 30";
        const auto result = runShell(script, 1.0, 0.3);
        check(result.timedOut, "tree kill: watchdog fired");

        const auto readMarker = [&]() -> std::string {
            std::ifstream file(marker);
            std::string value;
            if (file) { file >> value; }
            return value;
        };

        // Give any survivor time to advance the counter.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        const std::string first = readMarker();
        std::this_thread::sleep_for(std::chrono::milliseconds(700));
        const std::string second = readMarker();

        check(first == second,
              "tree kill: orphaned grandchild stopped writing (was '" + first +
                  "', now '" + second + "')");
        std::remove(marker.c_str());
    }

    // ---- peak memory ------------------------------------------------------
    {
        // Touches every page so the allocation is resident, not just mapped.
        const auto result = runShell(
            "python3 -c \"b=bytearray(200*1024*1024)\n"
            "for i in range(0,len(b),4096): b[i]=1\" 2>/dev/null || exit 0",
            60.0);
        if (result.exitCode == 0 && result.peakMemoryBytes.has_value()) {
            const double megabytes =
                static_cast<double>(*result.peakMemoryBytes) / (1024.0 * 1024.0);
            std::cout << "       measured peak RSS: " << megabytes << " MB\n";
            // Generous bounds: this asserts the number is real and in the right
            // units, not that the allocator behaved in a particular way. A
            // Darwin/Linux unit mix-up would land 1024x outside this window.
            check(megabytes > 100.0 && megabytes < 4096.0,
                  "peak memory: 200 MB allocation measured in a plausible range");
        } else {
            std::cout << "[SKIP] peak memory: python3 unavailable\n";
        }
    }

    // ---- output cap -------------------------------------------------------
    {
        benchmark::RunLimits limits;
        limits.wallClock = std::chrono::milliseconds(30000);
        limits.maxCapturedBytes = 4096;
        benchmark::ProcessManager manager;
        const auto result = manager.run(
            "/bin/sh", {"-c", "i=0; while [ $i -lt 2000 ]; do echo aaaaaaaaaaaaaaaaaaaa; i=$((i+1)); done"},
            limits);
        check(result.stdoutOutput.size() <= 4096, "output cap: stdout bounded");
        check(result.stdoutTruncated, "output cap: truncation reported");
        check(result.exitCode == 0,
              "output cap: child still finished (pipe was drained, not blocked)");
    }

    std::cout << (failures == 0 ? "\nAll process manager tests passed\n"
                                : "\nFAILURES\n");
    return failures == 0 ? 0 : 1;
}
