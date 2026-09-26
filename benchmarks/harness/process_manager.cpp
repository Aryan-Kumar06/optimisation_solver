#include "process_manager.h"

#if defined(_WIN32)
#error "benchmark::ProcessManager is POSIX-only; see process_manager.h."
#endif

#include <cerrno>
#include <csignal>
#include <cstring>
#include <fcntl.h>
#include <poll.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>

namespace benchmark {
namespace {

using Clock = std::chrono::steady_clock;

std::chrono::milliseconds elapsedSince(const Clock::time_point& start) {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - start);
}

void setNonBlocking(int descriptor) {
    const int flags = fcntl(descriptor, F_GETFL, 0);
    if (flags != -1) {
        fcntl(descriptor, F_SETFL, flags | O_NONBLOCK);
    }
}

// Appends up to the cap and reports whether anything was dropped.
void drain(int descriptor, std::string& sink, std::size_t cap, bool& truncated) {
    char buffer[8192];
    while (true) {
        const ssize_t bytesRead = ::read(descriptor, buffer, sizeof(buffer));
        if (bytesRead <= 0) {
            return;
        }
        if (sink.size() >= cap) {
            truncated = true;
            continue;  // keep draining so the child never blocks on a full pipe
        }
        const std::size_t room = cap - sink.size();
        const std::size_t take = std::min(room, static_cast<std::size_t>(bytesRead));
        sink.append(buffer, take);
        if (take < static_cast<std::size_t>(bytesRead)) {
            truncated = true;
        }
    }
}

// ru_maxrss is BYTES on Darwin and KILOBYTES on Linux. Getting this wrong is a
// silent 1024x error in a reported number, so it is decided at compile time
// rather than guessed from the magnitude.
std::int64_t maxRssToBytes(long value) {
#if defined(__APPLE__)
    return static_cast<std::int64_t>(value);
#else
    return static_cast<std::int64_t>(value) * 1024;
#endif
}

}  // namespace

ProcessResult ProcessManager::run(
    const std::string& executable,
    const std::vector<std::string>& arguments,
    const RunLimits& limits
) {
    ProcessResult result;

    int stdoutPipe[2];
    int stderrPipe[2];
    int execErrorPipe[2];

    if (::pipe(stdoutPipe) == -1) {
        result.launchFailed = true;
        result.launchError = "pipe(stdout): " + std::string(std::strerror(errno));
        return result;
    }
    if (::pipe(stderrPipe) == -1) {
        ::close(stdoutPipe[0]); ::close(stdoutPipe[1]);
        result.launchFailed = true;
        result.launchError = "pipe(stderr): " + std::string(std::strerror(errno));
        return result;
    }
    if (::pipe(execErrorPipe) == -1) {
        ::close(stdoutPipe[0]); ::close(stdoutPipe[1]);
        ::close(stderrPipe[0]); ::close(stderrPipe[1]);
        result.launchFailed = true;
        result.launchError = "pipe(execerr): " + std::string(std::strerror(errno));
        return result;
    }

    // Closed automatically by a successful exec(), which is how the parent
    // distinguishes "exec worked" from "exec failed and wrote an errno".
    const int flags = fcntl(execErrorPipe[1], F_GETFD);
    if (flags != -1) {
        fcntl(execErrorPipe[1], F_SETFD, flags | FD_CLOEXEC);
    }

    const Clock::time_point start = Clock::now();
    const pid_t pid = ::fork();

    if (pid == -1) {
        ::close(stdoutPipe[0]); ::close(stdoutPipe[1]);
        ::close(stderrPipe[0]); ::close(stderrPipe[1]);
        ::close(execErrorPipe[0]); ::close(execErrorPipe[1]);
        result.launchFailed = true;
        result.launchError = "fork: " + std::string(std::strerror(errno));
        return result;
    }

    // ---------------------------------------------------------------- child
    if (pid == 0) {
        ::close(stdoutPipe[0]);
        ::close(stderrPipe[0]);
        ::close(execErrorPipe[0]);

        const auto reportAndExit = [&](int code) {
            const int errorCode = code;
            ssize_t ignored = ::write(execErrorPipe[1], &errorCode, sizeof(errorCode));
            (void)ignored;
            _exit(127);
        };

        // Own process group, so the watchdog can signal this process and every
        // descendant it spawns with a single killpg().
        if (::setsid() == -1) {
            reportAndExit(errno);
        }
        if (::dup2(stdoutPipe[1], STDOUT_FILENO) == -1) {
            reportAndExit(errno);
        }
        if (::dup2(stderrPipe[1], STDERR_FILENO) == -1) {
            reportAndExit(errno);
        }
        ::close(stdoutPipe[1]);
        ::close(stderrPipe[1]);

        std::vector<char*> argv;
        argv.reserve(arguments.size() + 2);
        argv.push_back(const_cast<char*>(executable.c_str()));
        for (const std::string& argument : arguments) {
            argv.push_back(const_cast<char*>(argument.c_str()));
        }
        argv.push_back(nullptr);

        ::execvp(executable.c_str(), argv.data());
        reportAndExit(errno);
    }

    // --------------------------------------------------------------- parent
    ::close(stdoutPipe[1]);
    ::close(stderrPipe[1]);
    ::close(execErrorPipe[1]);

    setNonBlocking(stdoutPipe[0]);
    setNonBlocking(stderrPipe[0]);
    setNonBlocking(execErrorPipe[0]);

    bool finished = false;
    bool execFailureSeen = false;
    bool termSent = false;
    Clock::time_point termSentAt{};
    int waitStatus = 0;
    rusage usage{};

    while (!finished) {
        pollfd descriptors[2];
        descriptors[0].fd = stdoutPipe[0];
        descriptors[0].events = POLLIN;
        descriptors[1].fd = stderrPipe[0];
        descriptors[1].events = POLLIN;
        ::poll(descriptors, 2, 20);

        drain(stdoutPipe[0], result.stdoutOutput, limits.maxCapturedBytes,
              result.stdoutTruncated);
        drain(stderrPipe[0], result.stderrOutput, limits.maxCapturedBytes,
              result.stderrTruncated);

        if (!execFailureSeen) {
            int errorCode = 0;
            const ssize_t bytesRead =
                ::read(execErrorPipe[0], &errorCode, sizeof(errorCode));
            if (bytesRead == static_cast<ssize_t>(sizeof(errorCode))) {
                execFailureSeen = true;
                result.launchFailed = true;
                result.launchError = std::strerror(errorCode);
            }
        }

        // wait4 rather than waitpid: it fills rusage, so peak memory comes from
        // the kernel's own high-water mark instead of a sampling loop that can
        // step straight over a spike.
        const pid_t waited = ::wait4(pid, &waitStatus, WNOHANG, &usage);
        if (waited == pid) {
            finished = true;
            break;
        }

        if (limits.wallClock.count() > 0) {
            const auto elapsed = elapsedSince(start);
            if (!termSent && elapsed >= limits.wallClock) {
                result.timedOut = true;
                result.killedProcessTree = true;
                // Negative pid signals the whole group. The child called
                // setsid(), so its pid is its group id.
                ::kill(-pid, SIGTERM);
                termSent = true;
                termSentAt = Clock::now();
            } else if (termSent && elapsedSince(termSentAt) >= limits.gracePeriod) {
                result.escalatedToKill = true;
                ::kill(-pid, SIGKILL);
                // Reap synchronously; SIGKILL cannot be blocked.
                ::wait4(pid, &waitStatus, 0, &usage);
                finished = true;
            }
        }
    }

    drain(stdoutPipe[0], result.stdoutOutput, limits.maxCapturedBytes,
          result.stdoutTruncated);
    drain(stderrPipe[0], result.stderrOutput, limits.maxCapturedBytes,
          result.stderrTruncated);

    ::close(stdoutPipe[0]);
    ::close(stderrPipe[0]);
    ::close(execErrorPipe[0]);

    // Sweep the group even on a clean exit: the solve may have exited while a
    // forked helper it never reaped is still running, and leaving that behind
    // would contaminate the next instance's timing.
    ::kill(-pid, SIGKILL);

    if (usage.ru_maxrss > 0) {
        result.peakMemoryBytes = maxRssToBytes(usage.ru_maxrss);
    }

    if (!result.launchFailed) {
        if (WIFEXITED(waitStatus)) {
            result.exitCode = WEXITSTATUS(waitStatus);
        } else if (WIFSIGNALED(waitStatus)) {
            result.terminatingSignal = WTERMSIG(waitStatus);
            result.exitCode = 128 + WTERMSIG(waitStatus);
        }
    }

    result.runtime = elapsedSince(start);
    return result;
}

}  // namespace benchmark
