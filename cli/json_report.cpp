#include "json_report.h"

#include "solver/dispatcher.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <algorithm>
#include <vector>

namespace cli {
namespace {

// ---------------------------------------------------------------------------
// SHA-256. Self-contained on purpose: the instance hash ties a result row to
// the exact bytes it came from, and pulling in a crypto dependency for one
// digest would make the benchmark harness harder to build than the solver.
// ---------------------------------------------------------------------------

struct Sha256 {
    std::uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                              0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
    std::uint64_t bitCount = 0;
    std::array<unsigned char, 64> buffer{};
    std::size_t bufferLength = 0;

    static std::uint32_t rotr(std::uint32_t value, int bits) {
        return (value >> bits) | (value << (32 - bits));
    }

    void compress(const unsigned char* block) {
        static const std::uint32_t k[64] = {
            0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,
            0x923f82a4u,0xab1c5ed5u,0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,
            0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,0xe49b69c1u,0xefbe4786u,
            0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
            0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,
            0x06ca6351u,0x14292967u,0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,
            0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,0xa2bfe8a1u,0xa81a664bu,
            0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
            0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,
            0x5b9cca4fu,0x682e6ff3u,0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,
            0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};

        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (static_cast<std::uint32_t>(block[i * 4]) << 24) |
                   (static_cast<std::uint32_t>(block[i * 4 + 1]) << 16) |
                   (static_cast<std::uint32_t>(block[i * 4 + 2]) << 8) |
                   (static_cast<std::uint32_t>(block[i * 4 + 3]));
        }
        for (int i = 16; i < 64; ++i) {
            const std::uint32_t s0 =
                rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            const std::uint32_t s1 =
                rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        std::uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
        std::uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
        for (int i = 0; i < 64; ++i) {
            const std::uint32_t S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const std::uint32_t ch = (e & f) ^ (~e & g);
            const std::uint32_t t1 = h + S1 + ch + k[i] + w[i];
            const std::uint32_t S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const std::uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
            const std::uint32_t t2 = S0 + maj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        state[0] += a; state[1] += b; state[2] += c; state[3] += d;
        state[4] += e; state[5] += f; state[6] += g; state[7] += h;
    }

    void update(const unsigned char* data, std::size_t length) {
        bitCount += static_cast<std::uint64_t>(length) * 8;
        while (length > 0) {
            const std::size_t take = std::min(length, std::size_t{64} - bufferLength);
            std::copy(data, data + take, buffer.begin() + static_cast<std::ptrdiff_t>(bufferLength));
            bufferLength += take;
            data += take;
            length -= take;
            if (bufferLength == 64) {
                compress(buffer.data());
                bufferLength = 0;
            }
        }
    }

    std::string finish() {
        const std::uint64_t bits = bitCount;
        unsigned char pad = 0x80;
        update(&pad, 1);
        pad = 0x00;
        while (bufferLength != 56) {
            update(&pad, 1);
        }
        unsigned char lengthBytes[8];
        for (int i = 0; i < 8; ++i) {
            lengthBytes[i] = static_cast<unsigned char>((bits >> (56 - 8 * i)) & 0xff);
        }
        bitCount = bits;  // update() would otherwise count the length field itself
        std::copy(lengthBytes, lengthBytes + 8, buffer.begin() + 56);
        compress(buffer.data());

        std::ostringstream out;
        out << std::hex << std::setfill('0');
        for (std::uint32_t word : state) {
            out << std::setw(8) << word;
        }
        return out.str();
    }
};

// ---------------------------------------------------------------------------
// JSON emission
// ---------------------------------------------------------------------------

void writeString(std::ostream& out, const std::string& value) {
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

// JSON has no Infinity or NaN. Emitting them unquoted produces a document that
// strict parsers reject, and emitting 0 would be a lie, so both become null and
// the checker treats a missing bound as unbounded.
void writeNumber(std::ostream& out, double value) {
    if (!std::isfinite(value)) {
        out << "null";
        return;
    }
    // 17 significant digits round-trips an IEEE double exactly. A benchmark
    // that reported 6 digits could not tell a 1e-9 residual from a 1e-16 one.
    out << std::setprecision(17) << value;
}

void writeVector(std::ostream& out, const std::vector<double>& values) {
    out << '[';
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        writeNumber(out, values[i]);
    }
    out << ']';
}

void writeNames(std::ostream& out, const std::vector<std::string>& names) {
    out << '[';
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (i != 0) {
            out << ',';
        }
        writeString(out, names[i]);
    }
    out << ']';
}

}  // namespace

std::string sha256File(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    Sha256 digest;
    std::array<char, 65536> chunk{};
    while (file.read(chunk.data(), static_cast<std::streamsize>(chunk.size())) ||
           file.gcount() > 0) {
        digest.update(reinterpret_cast<const unsigned char*>(chunk.data()),
                      static_cast<std::size_t>(file.gcount()));
        if (!file) {
            break;
        }
    }
    return digest.finish();
}

bool writeModelDump(std::ostream& out, const model::Model& model) {
    out << "{\n  \"name\": ";
    writeString(out, model.name);
    out << ",\n  \"sense\": ";
    writeString(out, model.objective.sense == model::ObjectiveSense::Maximize
                         ? "max" : "min");
    out << ",\n  \"offset\": ";
    writeNumber(out, model.objective.offset);

    out << ",\n  \"variables\": [";
    for (std::size_t j = 0; j < model.variables.size(); ++j) {
        const model::Variable& variable = model.variables[j];
        if (j != 0) { out << ","; }
        out << "\n    {\"name\": ";
        writeString(out, variable.name);
        out << ", \"lower\": ";
        writeNumber(out, variable.lowerBound);
        out << ", \"upper\": ";
        writeNumber(out, variable.upperBound);
        out << ", \"type\": ";
        writeString(out, variable.type == model::VariableType::Binary    ? "B"
                       : variable.type == model::VariableType::Integer   ? "I"
                                                                         : "C");
        out << "}";
    }
    out << "\n  ],";

    out << "\n  \"objective_linear\": {";
    bool first = true;
    for (const auto& term : model.objective.linearTerms) {
        if (!first) { out << ", "; }
        first = false;
        out << "\"" << term.variableIndex << "\": ";
        writeNumber(out, term.value);
    }
    out << "},";

    // Emitted with i <= j so a triangular and a full-matrix source collapse to
    // the same key, which is what makes the two readers comparable at all.
    out << "\n  \"objective_quadratic\": {";
    first = true;
    for (const auto& term : model.objective.quadraticTerms) {
        const int lo = std::min(term.variableIndex1, term.variableIndex2);
        const int hi = std::max(term.variableIndex1, term.variableIndex2);
        if (!first) { out << ", "; }
        first = false;
        out << "\"" << lo << "," << hi << "\": ";
        writeNumber(out, term.value);
    }
    out << "},";

    out << "\n  \"constraints\": [";
    for (std::size_t i = 0; i < model.constraints.size(); ++i) {
        const model::Constraint& constraint = model.constraints[i];
        if (i != 0) { out << ","; }
        out << "\n    {\"name\": ";
        writeString(out, constraint.name);
        out << ", \"lower\": ";
        writeNumber(out, constraint.lowerBound);
        out << ", \"upper\": ";
        writeNumber(out, constraint.upperBound);
        out << ", \"terms\": {";
        bool firstTerm = true;
        for (const auto& term : constraint.linearTerms) {
            if (!firstTerm) { out << ", "; }
            firstTerm = false;
            out << "\"" << term.variableIndex << "\": ";
            writeNumber(out, term.value);
        }
        out << "}}";
    }
    out << "\n  ]\n}\n";
    return static_cast<bool>(out);
}

bool writeJsonReport(std::ostream& out,
                     const JsonReportInput& input,
                     const solver::SolveResult& result) {
    out << "{\n";

    out << "  \"schema\": \"optimsolver.solve.v1\",\n";

    out << "  \"instance\": {\"path\": ";
    writeString(out, input.instancePath);
    out << ", \"sha256\": ";
    if (input.instanceSha256.empty()) {
        out << "null";
    } else {
        writeString(out, input.instanceSha256);
    }
    out << ", \"variables\": " << input.originalVariables
        << ", \"constraints\": " << input.originalConstraints << "},\n";

    out << "  \"solver\": {\"name\": \"optimsolver\", \"commit\": ";
#ifdef OPTIMSOLVER_GIT_COMMIT
    writeString(out, OPTIMSOLVER_GIT_COMMIT);
#else
    out << "null";
#endif
    out << ", \"build_type\": ";
#ifdef OPTIMSOLVER_BUILD_TYPE
    writeString(out, OPTIMSOLVER_BUILD_TYPE);
#else
    out << "null";
#endif
    out << "},\n";

    // Requested is what the caller asked for; executed is what actually ran.
    // These are separate fields in SolveResult for a reason (they disagreed
    // once), so the record keeps them separate too.
    out << "  \"settings\": {\"requested_engine\": ";
    if (input.requestedEngine.empty()) {
        out << "null";
    } else {
        writeString(out, input.requestedEngine);
    }
    out << ", \"time_limit_seconds\": ";
    if (input.timeLimitSeconds > 0.0) {
        writeNumber(out, input.timeLimitSeconds);
    } else {
        out << "null";
    }
    out << ", \"tolerance\": ";
    writeNumber(out, input.tolerance);
    out << ", \"thread_count\": " << input.threadCount;
    out << "},\n";

    out << "  \"classification\": ";
    if (input.classification != nullptr) {
        out << "{\"problem_class\": ";
        writeString(out, solver::toString(input.classification->problemClass));
        out << ", \"num_binary\": " << input.classification->hints.numBinary
            << ", \"num_integer\": " << input.classification->hints.numInteger
            << ", \"num_continuous\": " << input.classification->hints.numContinuous
            << ", \"nonzeros\": " << input.classification->hints.numNonzeros
            << ", \"coef_range_ratio\": ";
        writeNumber(out, input.classification->hints.coefRangeRatio);
        out << "}";
    } else {
        out << "null";
    }
    out << ",\n";

    out << "  \"presolve\": ";
    if (input.presolve != nullptr) {
        out << "{\"infeasible\": " << (input.presolve->infeasible ? "true" : "false")
            << ", \"converged\": " << (input.presolve->converged ? "true" : "false")
            << ", \"reduced_variables\": " << input.presolve->presolvedVariables
            << ", \"reduced_constraints\": " << input.presolve->presolvedConstraints
            << ", \"transformations\": " << input.presolve->transformations.size() << "}";
    } else {
        out << "null";
    }
    out << ",\n";

    // The solver's own termination status. The harness keeps this strictly
    // separate from the independent checker's verdict: this field says what the
    // solver claims, never what was verified.
    out << "  \"termination\": {\"status\": ";
    writeString(out, solver::toString(result.status));
    out << ", \"message\": ";
    writeString(out, result.message);
    out << ", \"dispatched_engine\": ";
    writeString(out, solver::toString(result.engine));
    out << ", \"executed_engine\": ";
    if (result.executedEngine == solver::Engine::Unsupported) {
        // Unsupported here means no engine was invoked at all.
        out << "null";
    } else {
        writeString(out, solver::toString(result.executedEngine));
    }
    out << ", \"engine_reason\": ";
    writeString(out, result.engineReason);
    out << "},\n";

    // SolveResult now carries everything in the ORIGINAL model's coordinates:
    // presolve and postsolve run inside solver::solve(), so the record reads
    // from the result rather than from a postsolve pointer the CLI no longer
    // holds. hasPrimal is the single authority on whether a point exists --
    // an empty vector can be a COMPLETE solution for a model presolve
    // eliminated entirely, so emptiness must not be used as the test.
    const bool havePrimal = result.hasPrimal;

    out << "  \"objective\": ";
    if (havePrimal && std::isfinite(result.objectiveValue)) {
        writeNumber(out, result.objectiveValue);
    } else {
        out << "null";
    }
    out << ",\n";

    // Dual bound, gap and the branch-and-cut tree counters depend on the MILP
    // engine maintaining a global bound over open subtrees, which SolveResult
    // does not yet carry. The keys are emitted as null rather than omitted so
    // the schema keeps one shape: a consumer that has to distinguish "absent
    // key" from "null value" will eventually get that distinction wrong.
    out << "  \"dual_bound\": null,\n";
    out << "  \"mip_gap\": null,\n";

    out << "  \"primal\": ";
    if (havePrimal) {
        writeVector(out, result.variableValues);
    } else {
        out << "null";
    }
    out << ",\n";

    out << "  \"duals\": ";
    if (havePrimal && result.hasDuals) {
        writeVector(out, result.constraintDuals);
    } else {
        out << "null";
    }
    out << ",\n";

    out << "  \"reduced_costs\": ";
    if (havePrimal && result.hasDuals) {
        writeVector(out, result.reducedCosts);
    } else {
        out << "null";
    }
    out << ",\n";

    out << "  \"duals_unavailable_reason\": ";
    if (!result.hasDuals) {
        writeString(out, result.dualsUnavailableReason.empty()
                             ? std::string("engine produced no duals")
                             : result.dualsUnavailableReason);
    } else {
        out << "null";
    }
    out << ",\n";

    // Solver-reported residuals. These are the SOLVER's claim about its own
    // answer; the harness recomputes all of them independently.
    out << "  \"self_reported\": {\"max_bound_residual\": ";
    out << "null";  // residuals are recomputed independently by the checker
    out << ", \"max_constraint_residual\": ";
    out << "null";
    out << ", \"max_dual_residual\": ";
    if (result.hasDuals) {
        writeNumber(out, result.maxDualResidual);
    } else {
        out << "null";
    }
    out << ", \"max_integrality_violation\": ";
    writeNumber(out, result.maxIntegralityViolation);
    out << "},\n";

    // Stage breakdown. The process's end-to-end time lives in the run record
    // written by bench_runner; these are the parts of it the solver can see.
    out << "  \"stage_seconds\": {\"parse\": ";
    if (input.parseSeconds >= 0.0) { writeNumber(out, input.parseSeconds); } else { out << "null"; }
    out << ", \"presolve\": ";
    if (input.presolveSeconds >= 0.0) { writeNumber(out, input.presolveSeconds); } else { out << "null"; }
    out << ", \"solve\": ";
    if (input.solveSeconds >= 0.0) { writeNumber(out, input.solveSeconds); } else { out << "null"; }
    out << ", \"postsolve\": ";
    if (input.postsolveSeconds >= 0.0) { writeNumber(out, input.postsolveSeconds); } else { out << "null"; }
    out << "},\n";

    out << "  \"work\": {\"iterations\": ";
    if (result.iterations > 0) { out << result.iterations; } else { out << "null"; }
    out << ", \"nodes\": ";
    if (result.nodeCount > 0) { out << result.nodeCount; } else { out << "null"; }
    out << ", \"solve_seconds\": ";
    writeNumber(out, result.solveSeconds);
    out << "},\n";

    out << "  \"variable_names\": ";
    if (input.originalModel != nullptr) {
        std::vector<std::string> names;
        names.reserve(input.originalModel->variables.size());
        for (std::size_t j = 0; j < input.originalModel->variables.size(); ++j) {
            names.push_back(input.originalModel->variables[j].name.empty()
                                ? "x" + std::to_string(j)
                                : input.originalModel->variables[j].name);
        }
        writeNames(out, names);
    } else {
        out << "null";
    }
    out << ",\n";

    out << "  \"constraint_names\": ";
    if (input.originalModel != nullptr) {
        std::vector<std::string> names;
        names.reserve(input.originalModel->constraints.size());
        for (std::size_t i = 0; i < input.originalModel->constraints.size(); ++i) {
            names.push_back(input.originalModel->constraints[i].name.empty()
                                ? "c" + std::to_string(i)
                                : input.originalModel->constraints[i].name);
        }
        writeNames(out, names);
    } else {
        out << "null";
    }
    out << "\n}\n";

    return static_cast<bool>(out);
}

}  // namespace cli
