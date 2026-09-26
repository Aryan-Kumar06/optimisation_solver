#ifndef MPS_MPS_READER_H_
#define MPS_MPS_READER_H_

#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <limits>
#include "model/model.h"

namespace mps {

// Sections this reader understands. UNSUPPORTED is not a parse state: an
// unrecognised header is an error, never a skipped block.
//
// The reason is the failure mode it replaces. Before, an unrecognised header
// left current_section_ untouched, so a RANGES block arriving after RHS was
// parsed AS RHS data -- every range value silently overwrote a row's
// right-hand side. The file loaded, the solve converged, and the answer
// belonged to a different problem. Anything this reader cannot represent must
// therefore stop the parse rather than change the model behind the caller's
// back.
enum class MpsSection {
    NONE,
    NAME,
    OBJSENSE,
    ROWS,
    COLUMNS,
    RHS,
    RANGES,
    BOUNDS,
    QUADOBJ,   // QPS lower-triangular Hessian
    QMATRIX,   // QPS full Hessian
    ENDATA
};

// Internal tracker for row senses during parsing.
enum class RowSense {
    LESS_EQUAL,    // L
    GREATER_EQUAL, // G
    EQUAL          // E
};

// Parses fixed- and free-format MPS, and the QPS quadratic-objective
// extension, into a model::Model.
//
// Supported: NAME, OBJSENSE, ROWS, COLUMNS (including INTORG/INTEND integer
// markers), RHS (including an objective-row entry, which MPS defines as the
// NEGATED objective constant), RANGES, BOUNDS (LO UP FX FR MI PL BV LI UI),
// QUADOBJ, QMATRIX, ENDATA.
//
// Rejected with std::runtime_error, rather than skipped: SOS, INDICATORS,
// QCMATRIX/QCROWS, and any unrecognised section, row sense or bound type.
// Each of those changes the problem, and a solver that quietly drops them
// reports a confident optimum for a model the user did not write.
class MpsReader {
public:
    MpsReader() = default;

    // Throws std::runtime_error on an unreadable file or any construct that
    // cannot be represented faithfully.
    model::Model read(const std::string& filepath);

    // Non-fatal observations from the most recent read(): constructs that were
    // skipped without changing the represented problem (a second free row, an
    // RHS entry naming a row that does not exist), and conventions that were
    // applied where the format is ambiguous (a negative UP bound opening the
    // lower bound to -infinity).
    //
    // Empty after a clean parse. Surfacing these matters when many files are
    // read unattended, where nobody is watching an individual parse.
    [[nodiscard]] const std::vector<std::string>& warnings() const noexcept {
        return warnings_;
    }

private:
    MpsSection current_section_ = MpsSection::NONE;
    std::string objective_row_name_;

    // True while inside an INTORG/INTEND marker pair: columns first seen here
    // are integer.
    bool integer_marker_active_ = false;

    // Fast name-to-index mappings.
    std::unordered_map<std::string, size_t> var_name_to_idx_;
    std::unordered_map<std::string, size_t> constraint_name_to_idx_;

    // Row senses, and the RHS/RANGES values that pair with them. Bounds are
    // computed from all three at the end of the parse rather than as each
    // section is read, so a file that orders RANGES before RHS still produces
    // the same model.
    std::unordered_map<std::string, RowSense> row_senses_;
    // RHS and RANGES independently select the first named vector.
    std::string selected_rhs_vector_;
    std::string selected_ranges_vector_;
    std::vector<double> row_rhs_;
    std::vector<double> row_range_;
    std::vector<char> row_has_range_;

    // Free (type N) rows after the objective. Referenced coefficients are
    // dropped, which is what the format intends, but it is worth reporting.
    std::vector<std::string> free_row_names_;

    // Explicit lower bounds, tracked so the negative-UP convention can tell an
    // untouched default from a deliberate 0.
    std::vector<char> var_lower_explicit_;

    // Linear term accumulators (variable index -> coefficient).
    std::unordered_map<int, double> obj_term_map_;
    std::vector<std::unordered_map<int, double>> constraint_term_maps_;

    // Quadratic objective accumulator, keyed by (i, j) with i <= j so that a
    // full matrix and a triangular one collapse to the same representation.
    std::map<std::pair<int, int>, double> quad_term_map_;

    double objective_offset_ = 0.0;

    std::vector<std::string> warnings_;

    void reset();

    int get_or_create_variable(const std::string& var_name, model::Model& model);
    void add_coefficient(const std::string& var_name, const std::string& row_name,
                         double value, model::Model& model);
    void add_quadratic(const std::string& col1, const std::string& col2,
                       double value, bool mirror, model::Model& model);

    // Applies sense + RHS + RANGES to produce each row's final bound pair.
    void finalizeRowBounds(model::Model& model);

    void warn(const std::string& message);
};

} // namespace mps

#endif // MPS_MPS_READER_H_
