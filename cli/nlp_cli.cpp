#include "nlp_cli.h"
#include "json_report.h"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <stdexcept>

namespace cli {
namespace {
bool sameFile(const std::string& a, const std::string& b) {
    std::error_code ec;
    if (a == b || std::filesystem::equivalent(a, b, ec)) return true;
    // Also detect equivalent output paths when neither file exists yet.
    const auto left = std::filesystem::weakly_canonical(a, ec);
    if (ec) return false;
    const auto right = std::filesystem::weakly_canonical(b, ec);
    return !ec && left == right;
}
}
int runNlp(const SolveOptions& options, std::ostream& out, std::ostream& err,
           const nlp::Input* loaded, solver::NlpSolveResult* stored) {
    try {
        if (options.dumpModelPath || options.threadCount)
            throw std::invalid_argument("--dump-model and --threads are not supported for NLP");
        const auto requested = options.solver ? solver::parseEngine(*options.solver) : std::optional<solver::Engine>{};
        if (options.solver && (!requested || *requested != solver::Engine::Nlp))
            throw std::invalid_argument("a nonlinear model requires --solver nlp (or automatic selection)");
        for (const auto* path : {&options.jsonPath, &options.outputPath})
            if (*path && sameFile(**path, options.modelPath))
                throw std::invalid_argument("output must differ from input path");
        if (options.jsonPath && options.outputPath && sameFile(*options.jsonPath, *options.outputPath))
            throw std::invalid_argument("JSON and solution output paths must differ");

        auto start = std::chrono::steady_clock::now();
        std::optional<nlp::Input> parsed;
        if (!loaded) {
            std::ifstream file(options.modelPath);
            if (!file) throw std::invalid_argument("cannot open NLP model");
            parsed.emplace(nlp::read(file));
            loaded = &*parsed;
        }
        nlp::Options numerical;
        if (options.tolerance) numerical.tolerance = *options.tolerance;
        if (options.iterationLimit) numerical.iterationLimit = *options.iterationLimit;
        if (options.timeLimitSeconds) numerical.timeLimitSeconds = *options.timeLimitSeconds;

        JsonReportInput metadata;
        metadata.instancePath = options.modelPath;
        // A cached interactive model can differ from the current disk file.
        if (parsed) metadata.instanceSha256 = sha256File(options.modelPath);
        metadata.requestedEngine = options.solver.value_or("");
        metadata.requestedBackend = options.backend.value_or("auto");
        metadata.cudaDevice = options.cudaDevice.value_or(0);
        metadata.tolerance = numerical.tolerance;
        metadata.timeLimitSeconds = numerical.timeLimitSeconds;
        metadata.originalVariables = loaded->model.variableBounds().size();
        metadata.originalConstraints = loaded->model.constraintBounds().size();
        metadata.parseSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
        const auto result = solver::solve(loaded->model, loaded->initial, numerical, requested);
        if (stored) *stored = result;
        // NLP has no CUDA backend; acknowledge a CUDA request on stderr, as the
        // orchestrator does for its CPU-only engines (stdout is the JSON report).
        if (options.backend && solver::parseComputeBackend(*options.backend) == solver::ComputeBackend::Cuda)
            err << "Compute backend: cpu (" << solver::toString(solver::Engine::Nlp)
                << " has no CUDA backend; ran on the CPU)\n";
        if (!writeJsonReport(out, metadata, result)) throw std::runtime_error("cannot write NLP report");
        if (options.jsonPath) {
            std::ofstream file(*options.jsonPath);
            if (!file || !writeJsonReport(file, metadata, result)) throw std::runtime_error("cannot write JSON output");
            file.close();
            if (!file) throw std::runtime_error("cannot finish JSON output");
        }
        if (options.outputPath) {
            std::ofstream file(*options.outputPath);
            if (!file) throw std::runtime_error("cannot open solution output");
            file << "Status: " << nlp::toString(result.status) << "\n"
                 << "Feasible: " << (result.feasible ? "yes" : "no") << "\n"
                 << "First-order local solver; no global optimality certificate.\n";
            if (result.hasPrimal) {
                file << std::setprecision(17) << "Objective: " << result.objective << "\n";
                for (size_t j = 0; j < result.primal.size(); ++j) file << "x[" << j << "] = " << result.primal[j] << "\n";
            }
            file.close();
            if (!file) throw std::runtime_error("cannot write solution output");
        }
        return result.status == nlp::Status::FirstOrderStationary ? 0 : 2;
    } catch (const std::exception& e) {
        err << "NLP input/output error: " << e.what() << '\n';
        return 1;
    }
}
} // namespace cli
