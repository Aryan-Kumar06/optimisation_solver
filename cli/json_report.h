#pragma once

// Machine-readable record of one solve, for the benchmark harness.
//
// This is deliberately a separate output path from the dashboard and from
// --output. The dashboard is for a person, --output is a solution listing, and
// this is the only one an automated comparison should ever parse: it carries
// the fields a scored run needs (requested vs executed engine, settings,
// status, objective, primal, duals, work counters) with a stable schema.
//
// EVERY value is emitted in the ORIGINAL model's coordinates, after postsolve.
// A benchmark that compared reduced-space vectors against a reference solver
// would be comparing two different problems.
//
// A value that does not exist is emitted as JSON null, never as 0.0 or an
// empty string. "Branch-and-cut produced no duals" and "the duals are all
// zero" are different facts, and a checker must be able to tell them apart.

#include "model/model.h"
#include "presolve/presolve_result.h"
#include "solver/classifier.h"
#include "solver/solve_result.h"

#include <iosfwd>
#include <string>

namespace cli {

// Everything the report needs that SolveResult does not already carry.
struct JsonReportInput {
    std::string instancePath;
    std::string instanceSha256;

    // What the caller asked for, as opposed to what the dispatcher chose.
    std::string requestedEngine;   // empty when the caller forced nothing
    double timeLimitSeconds = 0.0; // 0 means no limit
    double tolerance = 0.0;

    std::size_t originalVariables = 0;
    std::size_t originalConstraints = 0;

    const solver::Classification* classification = nullptr;
    const presolve::PresolveResult* presolve = nullptr;

    const model::Model* originalModel = nullptr;

    // Wall-clock seconds per stage, measured by the CLI around each call.
    // Reported separately from the process's end-to-end time so a slow run can
    // be attributed: a parse that dominates and an engine that dominates are
    // different problems with different fixes. Negative means "not run".
    double parseSeconds = -1.0;
    double presolveSeconds = -1.0;
    double solveSeconds = -1.0;      // orchestrator: dispatch + engine
    double postsolveSeconds = -1.0;

    int threadCount = 0;
};

// Writes the record. Returns false if the stream went bad while writing.
bool writeJsonReport(std::ostream& out,
                     const JsonReportInput& input,
                     const solver::SolveResult& result);

// Dumps the parsed model exactly as model::Model holds it, so an independent
// reader can be diffed against it field by field. This is the only way to test
// the parser's OUTPUT rather than its effect on an answer: a sense flip, a
// dropped objective constant or a mis-signed range can all produce a solve that
// looks entirely healthy while describing a different problem.
bool writeModelDump(std::ostream& out, const model::Model& model);

// SHA-256 of the instance file's bytes, so a result can be tied to the exact
// input it came from. Empty string if the file cannot be read.
[[nodiscard]] std::string sha256File(const std::string& path);

}  // namespace cli
