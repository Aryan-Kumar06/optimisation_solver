#pragma once
#include "argument_parser.h"
#include "solver/nlp.h"
#include <iosfwd>
namespace cli {
// Interactive callers pass their already-loaded snapshot; batch callers load
// once here. Both paths use the same orchestrator and reporting contract.
int runNlp(const SolveOptions& options, std::ostream& out, std::ostream& err,
           const nlp::Input* loaded = nullptr, solver::NlpSolveResult* stored = nullptr);
}
