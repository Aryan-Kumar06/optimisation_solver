#include <iostream>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <string>
#include <cstdlib>
#include "mps/mps_reader.h"

void test_basic_lp() {
    mps::MpsReader reader;
    model::Model m = reader.read("tests/mps/test_cases/01_basic_lp.mps");
    assert(m.variables.size() == 2);
    assert(m.constraints.size() == 3);

    // Objective linear terms checking
    assert(m.objective.linearTerms.size() == 2);

    // Constraint senses check (L, G, E)
    // L: [-inf, 10.0]
    assert(std::isinf(m.constraints[0].lowerBound) && m.constraints[0].lowerBound < 0);
    assert(m.constraints[0].upperBound == 10.0);

    // G: [5.0, +inf]
    assert(m.constraints[1].lowerBound == 5.0);
    assert(std::isinf(m.constraints[1].upperBound) && m.constraints[1].upperBound > 0);

    // E: [7.0, 7.0]
    assert(m.constraints[2].lowerBound == 7.0);
    assert(m.constraints[2].upperBound == 7.0);
    std::cout << "[PASS] Test 1: Basic LP (L, G, E)" << std::endl;
}

void test_multi_column_coeffs() {
    mps::MpsReader reader;
    model::Model m = reader.read("tests/mps/test_cases/02_multi_column_coeffs.mps");

    assert(m.variables.size() == 2);
    assert(m.constraints.size() == 2);

    // Check multiple coefficients on single COLUMNS line
    assert(m.constraints[0].linearTerms.size() == 1);
    assert(m.constraints[0].linearTerms[0].value == 2.5);
    std::cout << "[PASS] Test 2: Multi-column Coefficients" << std::endl;
}

void test_all_bound_types() {
    mps::MpsReader reader;
    model::Model m = reader.read("tests/mps/test_cases/03_all_bound_types.mps");

    // LO: [2.5, +inf]
    assert(m.variables[0].lowerBound == 2.5);
    assert(std::isinf(m.variables[0].upperBound));

    // UP: [0.0, 10.0] (default LO=0)
    assert(m.variables[1].lowerBound == 0.0);
    assert(m.variables[1].upperBound == 10.0);

    // FX: [7.0, 7.0]
    assert(m.variables[2].lowerBound == 7.0);
    assert(m.variables[2].upperBound == 7.0);

    // FR: [-inf, +inf]
    assert(std::isinf(m.variables[3].lowerBound) && m.variables[3].lowerBound < 0);
    assert(std::isinf(m.variables[3].upperBound) && m.variables[3].upperBound > 0);
    std::cout << "[PASS] Test 3: All Bound Types (LO, UP, FX, FR)" << std::endl;
}

void test_bounds_only_variable() {
    mps::MpsReader reader;
    model::Model m = reader.read("tests/mps/test_cases/04_bounds_only_variable.mps");

    // X_UNSEEN appeared only in BOUNDS section
    assert(m.variables.size() == 2);
    bool found_unseen = false;
    for (const auto& var : m.variables) {
        if (var.name == "X_UNSEEN") {
            found_unseen = true;
            assert(var.lowerBound == 3.0);
            assert(var.upperBound == 8.0);
        }
    }
    assert(found_unseen);
    std::cout << "[PASS] Test 4: Variable appearing only in BOUNDS" << std::endl;
}

void test_negatives_comments_blanks() {
    mps::MpsReader reader;
    model::Model m = reader.read("tests/mps/test_cases/05_negatives_comments_blanks.mps");

    assert(m.name == "NEG_COMMENTS");
    assert(m.objective.linearTerms.size() == 2);

    // Find the objective term for variable X1
    bool found_x1 = false;
    for (const auto& term : m.objective.linearTerms) {
        // If your linearTerm uses varIndex, check against m.variables[term.varIndex].name
        // Or if it stores varName directly, check term.varName == "X1"
        if (m.variables[term.variableIndex].name == "X1") {
            assert(std::abs(term.value - (-5.25)) < 1e-6);
            found_x1 = true;
        }
    }
    assert(found_x1);

    // Check constraint RHS negative value
    assert(m.constraints[0].upperBound == -15.0);
    std::cout << "[PASS] Test 5: Negative coefficients, comments, and blank lines" << std::endl;
}

void test_malformed_missing_section() {
    mps::MpsReader reader;
    // Should parse without crashing, returning an empty/partial model IR safely
    model::Model m = reader.read("tests/mps/test_cases/06_malformed_missing_section.mps");
    assert(m.name == "MALFORMED");
    assert(m.constraints.empty());
    std::cout << "[PASS] Test 6: Malformed / missing section fallback" << std::endl;
}


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static const model::Variable& varNamed(const model::Model& m, const std::string& name) {
    for (const auto& v : m.variables) {
        if (v.name == name) return v;
    }
    std::cerr << "no variable named " << name << std::endl;
    std::abort();
}

static const model::Constraint& rowNamed(const model::Model& m, const std::string& name) {
    for (const auto& c : m.constraints) {
        if (c.name == name) return c;
    }
    std::cerr << "no constraint named " << name << std::endl;
    std::abort();
}

static double quadCoefficient(const model::Model& m, const std::string& a,
                              const std::string& b) {
    int ia = -1, ib = -1;
    for (std::size_t j = 0; j < m.variables.size(); ++j) {
        if (m.variables[j].name == a) ia = static_cast<int>(j);
        if (m.variables[j].name == b) ib = static_cast<int>(j);
    }
    double total = 0.0;
    for (const auto& t : m.objective.quadraticTerms) {
        const bool hit = (t.variableIndex1 == ia && t.variableIndex2 == ib) ||
                         (t.variableIndex1 == ib && t.variableIndex2 == ia);
        if (hit) total += t.value;
    }
    return total;
}

static bool readThrows(const std::string& path, std::string& message) {
    mps::MpsReader reader;
    try {
        reader.read(path);
    } catch (const std::exception& ex) {
        message = ex.what();
        return true;
    }
    return false;
}

// ---------------------------------------------------------------------------
// RANGES
// ---------------------------------------------------------------------------

// The regression this whole batch exists for. Before RANGES was a known
// section, its header left current_section_ at RHS and every range value was
// consumed as a right-hand side -- so C1's rhs became 4.0 instead of 10.0 and
// the file loaded without complaint.
void test_ranges() {
    mps::MpsReader reader;
    model::Model m = reader.read("tests/mps/test_cases/07_ranges.mps");

    // L row, rhs 10, range 4 -> [rhs - |R|, rhs]
    assert(rowNamed(m, "C1").lowerBound == 6.0);
    assert(rowNamed(m, "C1").upperBound == 10.0);

    // G row, rhs 5, range 4 -> [rhs, rhs + |R|]
    assert(rowNamed(m, "C2").lowerBound == 5.0);
    assert(rowNamed(m, "C2").upperBound == 9.0);

    // E row, rhs 7, range +3 -> [rhs, rhs + R]
    assert(rowNamed(m, "C3").lowerBound == 7.0);
    assert(rowNamed(m, "C3").upperBound == 10.0);

    // E row, rhs 7, range -3 -> [rhs + R, rhs]. The sign of R choosing the
    // side is the part of the spec readers most often get wrong.
    assert(rowNamed(m, "C4").lowerBound == 4.0);
    assert(rowNamed(m, "C4").upperBound == 7.0);

    std::cout << "[PASS] Test 7: RANGES on L, G and both signs of E" << std::endl;
}

// ---------------------------------------------------------------------------
// Integer markers
// ---------------------------------------------------------------------------

void test_integer_markers() {
    mps::MpsReader reader;
    model::Model m = reader.read("tests/mps/test_cases/08_integer_markers.mps");

    assert(m.variables.size() == 4);
    assert(varNamed(m, "XCONT").type == model::VariableType::Continuous);
    assert(varNamed(m, "XINT1").type == model::VariableType::Integer);
    assert(varNamed(m, "XINT2").type == model::VariableType::Integer);
    // INTEND must close the block, or every later column becomes integer.
    assert(varNamed(m, "XCONT2").type == model::VariableType::Continuous);

    // A general integer defaults to [0, +inf), not [0, 1]: the older default
    // silently turns general integers into binaries.
    assert(varNamed(m, "XINT1").lowerBound == 0.0);
    assert(std::isinf(varNamed(m, "XINT1").upperBound));

    std::cout << "[PASS] Test 8: INTORG/INTEND integer markers" << std::endl;
}

// ---------------------------------------------------------------------------
// Extended bound types
// ---------------------------------------------------------------------------

void test_extended_bounds() {
    mps::MpsReader reader;
    model::Model m = reader.read("tests/mps/test_cases/09_extended_bounds.mps");

    // MI opens the lower bound and leaves the upper alone; the later UP then
    // closes it.
    assert(std::isinf(varNamed(m, "X_MI").lowerBound) &&
           varNamed(m, "X_MI").lowerBound < 0);
    assert(varNamed(m, "X_MI").upperBound == 4.0);

    assert(std::isinf(varNamed(m, "X_PL").upperBound) &&
           varNamed(m, "X_PL").upperBound > 0);

    assert(varNamed(m, "X_BV").type == model::VariableType::Binary);
    assert(varNamed(m, "X_BV").lowerBound == 0.0);
    assert(varNamed(m, "X_BV").upperBound == 1.0);

    assert(varNamed(m, "X_LI").type == model::VariableType::Integer);
    assert(varNamed(m, "X_LI").lowerBound == 2.0);

    assert(varNamed(m, "X_UI").type == model::VariableType::Integer);
    assert(varNamed(m, "X_UI").upperBound == 9.0);

    // A negative UP against an implicit lower bound of 0 would otherwise give
    // lb > ub and fail validation outright.
    assert(varNamed(m, "X_NEGUP").upperBound == -3.0);
    assert(std::isinf(varNamed(m, "X_NEGUP").lowerBound) &&
           varNamed(m, "X_NEGUP").lowerBound < 0);
    assert(m.validate());

    bool warnedAboutNegativeUp = false;
    for (const auto& w : reader.warnings()) {
        if (w.find("X_NEGUP") != std::string::npos) warnedAboutNegativeUp = true;
    }
    assert(warnedAboutNegativeUp);

    std::cout << "[PASS] Test 9: MI, PL, BV, LI, UI and negative UP" << std::endl;
}

// ---------------------------------------------------------------------------
// OBJSENSE
// ---------------------------------------------------------------------------

void test_objsense() {
    mps::MpsReader reader;

    model::Model block = reader.read("tests/mps/test_cases/10_objsense_max.mps");
    assert(block.objective.sense == model::ObjectiveSense::Maximize);

    // Both spellings occur in the wild: the sense on its own indented line,
    // and the sense on the header line itself.
    model::Model inlineForm = reader.read("tests/mps/test_cases/11_objsense_inline.mps");
    assert(inlineForm.objective.sense == model::ObjectiveSense::Maximize);

    // A file with no OBJSENSE still minimises.
    model::Model plain = reader.read("tests/mps/test_cases/01_basic_lp.mps");
    assert(plain.objective.sense == model::ObjectiveSense::Minimize);

    std::cout << "[PASS] Test 10: OBJSENSE, both spellings, default preserved"
              << std::endl;
}

// ---------------------------------------------------------------------------
// Objective constant
// ---------------------------------------------------------------------------

void test_objective_constant() {
    mps::MpsReader reader;
    model::Model m = reader.read("tests/mps/test_cases/12_objective_constant.mps");

    // MPS defines the objective-row RHS as the NEGATED constant, so -25.0 in
    // the file means +25.0 in the model. Dropping it shifts every reported
    // objective; adding it unnegated flips the shift's sign.
    assert(std::abs(m.objective.offset - 25.0) < 1e-9);

    // The entry must not also land in the constraint list.
    assert(m.constraints.size() == 1);
    assert(rowNamed(m, "C1").lowerBound == 4.0);

    std::cout << "[PASS] Test 11: objective constant from an RHS on the N row"
              << std::endl;
}

// ---------------------------------------------------------------------------
// Quadratic objective
// ---------------------------------------------------------------------------

// Both files describe 0.5 * x'Qx with Q = [[4,2],[2,4]], one as a lower
// triangle and one in full. model::Model stores the DIRECT coefficient with no
// implicit 0.5, so the expected objective is 2x^2 + 2xy + 2y^2 in both cases.
void test_quadratic_objective() {
    mps::MpsReader reader;

    model::Model lower = reader.read("tests/mps/test_cases/13_quadobj.mps");
    assert(std::abs(quadCoefficient(lower, "X", "X") - 2.0) < 1e-9);
    assert(std::abs(quadCoefficient(lower, "X", "Y") - 2.0) < 1e-9);
    assert(std::abs(quadCoefficient(lower, "Y", "Y") - 2.0) < 1e-9);

    model::Model full = reader.read("tests/mps/test_cases/14_qmatrix.mps");
    assert(std::abs(quadCoefficient(full, "X", "X") - 2.0) < 1e-9);
    assert(std::abs(quadCoefficient(full, "X", "Y") - 2.0) < 1e-9);
    assert(std::abs(quadCoefficient(full, "Y", "Y") - 2.0) < 1e-9);

    // The two spellings must produce the same model. Halving both formats --
    // or neither -- is wrong by a factor of two in exactly one of them, and
    // nothing in the status would show it.
    assert(quadCoefficient(lower, "X", "Y") == quadCoefficient(full, "X", "Y"));

    // An off-diagonal is stored once, not mirrored into two IR terms.
    assert(lower.objective.quadraticTerms.size() == 3);
    assert(full.objective.quadraticTerms.size() == 3);

    std::cout << "[PASS] Test 12: QUADOBJ and QMATRIX agree" << std::endl;
}

// ---------------------------------------------------------------------------
// Rejection rather than silent mis-parse
// ---------------------------------------------------------------------------

void test_rejects_unknown_section() {
    std::string message;
    assert(readThrows("tests/mps/test_cases/15_unknown_section.mps", message));
    assert(message.find("NOTASECTION") != std::string::npos);
    std::cout << "[PASS] Test 13: unrecognised section is rejected, not absorbed"
              << std::endl;
}

void test_rejects_sos() {
    std::string message;
    assert(readThrows("tests/mps/test_cases/16_sos_section.mps", message));
    assert(message.find("SOS") != std::string::npos);
    std::cout << "[PASS] Test 14: SOS section is rejected rather than dropped"
              << std::endl;
}

// Use runtime checks so these regressions also run in Release builds.
static void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void test_first_vectors() {
    mps::MpsReader reader;
    auto m = reader.read("tests/mps/test_cases/17_multiple_rhs.mps");
    require(rowNamed(m, "C1").upperBound == 10, "later RHS overwrote first vector");
    require(rowNamed(m, "C2").lowerBound == 5, "first RHS continuation was ignored");
    require(rowNamed(m, "C3").upperBound == 0, "later RHS supplied an absent entry");
    require(m.objective.offset == 25, "later RHS overwrote objective constant");
    require(!reader.warnings().empty(), "ignored RHS vectors need a warning");

    // Reuse the reader: vector selection must reset, and the first RANGES name
    // is independent of the first RHS name even when RANGES comes first.
    m = reader.read("tests/mps/test_cases/18_multiple_ranges.mps");
    require(rowNamed(m, "C1").lowerBound == 6 && rowNamed(m, "C1").upperBound == 10,
            "first RANGES/RHS vectors must determine the L row");
    require(rowNamed(m, "C2").lowerBound == 5 && rowNamed(m, "C2").upperBound == 8,
            "first RANGES continuation must determine the G row");
    require(std::isinf(rowNamed(m, "C3").lowerBound) &&
            rowNamed(m, "C3").upperBound == 8, "later RANGES supplied an absent entry");
    bool warnedRanges = false;
    for (const auto& warning : reader.warnings())
        warnedRanges |= warning.find("RANGES vector 'RANGE_B'") != std::string::npos;
    require(warnedRanges, "ignored RANGES vectors need a warning");
    m = reader.read("tests/mps/test_cases/07_ranges.mps");
    require(rowNamed(m, "C1").lowerBound == 6, "RANGES selection did not reset");
    std::cout << "[PASS] First RHS/RANGES vectors, independent selection and reader reset\n";
}

void test_marker_fields() {
    mps::MpsReader reader;
    const auto m = reader.read("tests/mps/test_cases/19_marker_names.mps");
    require(m.variables.size() == 5, "marker-like variable names were consumed as records");
    for (const auto& name : {"X_MARKER_INTORG", "MARKER", "X", "CONT_MARKER_INTORG"})
        require(varNamed(m, name).type == model::VariableType::Continuous,
                std::string("ordinary column changed marker state: ") + name);
    require(varNamed(m, "INTEGER_MARKER_INTEND").type == model::VariableType::Integer,
            "quoted marker fields must control integrality");
    require(rowNamed(m, "C1").linearTerms.size() == 4, "column coefficients were dropped");
    require(rowNamed(m, "MARKER").linearTerms.size() == 1 &&
            rowNamed(m, "MARKER").linearTerms[0].value == 5,
            "a row named MARKER with a numeric coefficient is not a marker record");
    std::cout << "[PASS] Marker detection uses fields, not substrings\n";
}

void test_integral_bound_values() {
    mps::MpsReader reader;
    const auto m = reader.read("tests/mps/test_cases/22_integral_bounds.mps");
    require(varNamed(m, "X").type == model::VariableType::Integer &&
            varNamed(m, "X").lowerBound == -2 && varNamed(m, "X").upperBound == 9,
            "integral LI/UI bounds must accept signed, decimal and exponent notation");
    require(varNamed(m, "Y").type == model::VariableType::Integer &&
            varNamed(m, "Y").lowerBound == 0 && varNamed(m, "Y").upperBound == 0,
            "zero LI/UI bounds are integral");
    for (const auto& bound : {std::make_pair("20_fractional_li.mps", "LI"),
                              std::make_pair("21_fractional_ui.mps", "UI")}) {
        std::string message;
        require(readThrows(std::string("tests/mps/test_cases/") + bound.first, message),
                std::string("fractional ") + bound.second + " was accepted");
        require(message.find("line 7") != std::string::npos &&
                message.find(bound.second) != std::string::npos &&
                message.find("integral value") != std::string::npos,
                "fractional bound error must identify its line and bound type: " + message);
    }
    std::cout << "[PASS] LI/UI require integral values with line-numbered errors\n";
}

// The RHS/RANGES set name is optional. Netlib's blend omits it, and before
// this the reader consumed field 1 as the set name regardless, took a row name
// for it, and failed the whole file. Parity of the field count is what decides:
// records are (row, value) pairs, so an odd count carries a name and an even
// count does not.
void test_rhs_without_set_name() {
    mps::MpsReader reader;
    model::Model m = reader.read("tests/mps/test_cases/17_rhs_no_set_name.mps");

    assert(std::isinf(rowNamed(m, "C1").lowerBound) &&
           rowNamed(m, "C1").lowerBound < 0);
    assert(rowNamed(m, "C1").upperBound == 10.0);

    // RANGES also omits its set name; G row with rhs 5 and range 4 -> [5, 9].
    assert(rowNamed(m, "C2").lowerBound == 5.0);
    assert(rowNamed(m, "C2").upperBound == 9.0);

    // A named set must still work, and must still select the first vector.
    model::Model named = reader.read("tests/mps/test_cases/07_ranges.mps");
    assert(rowNamed(named, "C1").lowerBound == 6.0);
    assert(rowNamed(named, "C1").upperBound == 10.0);

    std::cout << "[PASS] Test 15: RHS/RANGES with the set name omitted" << std::endl;
}

int main() {
    std::cout << "--- Running MPS Reader Test Suite ---" << std::endl;
    test_basic_lp();
    test_multi_column_coeffs();
    test_all_bound_types();
    test_bounds_only_variable();
    test_negatives_comments_blanks();
    test_malformed_missing_section();
    test_ranges();
    test_integer_markers();
    test_extended_bounds();
    test_objsense();
    test_objective_constant();
    test_quadratic_objective();
    test_rejects_unknown_section();
    test_rejects_sos();
    test_rhs_without_set_name();
    test_first_vectors();
    test_marker_fields();
    test_integral_bound_values();
    std::cout << "All MPS parser tests completed successfully!" << std::endl;
    return 0;
}