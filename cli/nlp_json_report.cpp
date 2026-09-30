#include "json_report.h"
#include <cmath>
#include <iomanip>
#include <ostream>
namespace cli {
namespace {
void string(std::ostream& out,const std::string& s) {
    out<<'"';
    for(unsigned char c:s) {
        if(c=='"'||c=='\\') out<<'\\'<<c;
        else if(c<32) out<<"\\u00"<<"0123456789abcdef"[c/16]<<"0123456789abcdef"[c%16];
        else out<<c;
    }
    out<<'"';
}
void number(std::ostream& out,double v) {if(std::isfinite(v)) out<<std::defaultfloat<<std::setprecision(17)<<v;else out<<"null";}
void vector(std::ostream& out,const std::vector<double>& v) {out<<'[';for(size_t i=0;i<v.size();++i) {if(i) out<<',';number(out,v[i]);}out<<']';}
} // namespace
bool writeJsonReport(std::ostream& out, const JsonReportInput& input, const solver::NlpSolveResult& r) {
    out << "{\"schema\":\"optimsolver.nlp.v1\",\"problem_class\":\"NLP\",\"engine\":";
    string(out, solver::toString(r.engine));
    out << ",\"executed_engine\":"; string(out, solver::toString(r.executedEngine));
    out << ",\"engine_reason\":"; string(out, r.engineReason);
    out << ",\"requested_engine\":"; string(out, input.requestedEngine);
    out << ",\"compute_backend\":";
    const bool ran = r.executedEngine == solver::Engine::Nlp && r.status != nlp::Status::InvalidProblem;
    writeBackendReport(out, input, ran ? "cpu" : nullptr,
        ran ? "nlp_sqp has no CUDA backend; ran on the CPU" : r.message);
    out << ",\"instance_path\":"; string(out, input.instancePath);
    out << ",\"instance_sha256\":"; string(out, input.instanceSha256);
    out << ",\"original_variables\":" << input.originalVariables;
    out << ",\"original_constraints\":" << input.originalConstraints;
    out << ",\"presolve_applied\":false,\"postsolve_applied\":false";
    out << ",\"tolerance\":"; number(out, input.tolerance);
    out << ",\"time_limit_seconds\":"; number(out, input.timeLimitSeconds);
    out << ",\"parse_seconds\":"; number(out, input.parseSeconds);
    out << ",\"status\":"; string(out,nlp::toString(r.status));
    out<<",\"message\":";string(out,r.message);
    out<<",\"has_primal\":"<<(r.hasPrimal?"true":"false")<<",\"feasible\":"<<(r.feasible?"true":"false");
    out<<",\"objective\":";number(out,r.hasPrimal?r.objective:nlp::infinity);
    out<<",\"primal_residual\":";number(out,r.primalResidual);
    out<<",\"dual_residual\":";number(out,r.dualResidual);
    out<<",\"complementarity\":";number(out,r.complementarity);
    out<<",\"iterations\":"<<r.iteration<<",\"qp_iterations\":"<<r.qpIterations<<",\"evaluations\":"<<r.evaluations;
    out<<",\"elastic_subproblems\":"<<r.elasticSubproblems<<",\"rejected_trials\":"<<r.rejectedTrials;
    out<<",\"solve_seconds\":";number(out,r.solveSeconds);
    out<<",\"primal\":";vector(out,r.primal);
    out<<",\"constraint_multipliers\":";vector(out,r.constraintMultipliers);
    out<<",\"bound_multipliers\":";vector(out,r.boundMultipliers);out<<"}\n";
    return static_cast<bool>(out);
}
} // namespace cli
