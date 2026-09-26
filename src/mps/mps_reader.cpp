#include "mps/mps_reader.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace mps {
namespace {

constexpr double kInfinity = std::numeric_limits<double>::infinity();

std::string toUpper(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
        return static_cast<char>(std::toupper(c));
    });
    return value;
}

// MPS marker records quote their keywords: 'MARKER' and 'INTORG'/'INTEND'.
// Strip them so the comparison does not depend on the writer's quoting style.
std::string unquote(std::string value) {
    value.erase(std::remove(value.begin(), value.end(), '\''), value.end());
    value.erase(std::remove(value.begin(), value.end(), '"'), value.end());
    return value;
}

std::vector<std::string> splitFields(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream stream(line);
    std::string token;
    while (stream >> token) {
        fields.push_back(token);
    }
    return fields;
}

bool parseNumber(const std::string& token, double& out) {
    try {
        std::size_t consumed = 0;
        const double value = std::stod(token, &consumed);
        if (consumed != token.size()) {
            return false;
        }
        out = value;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

[[noreturn]] void fail(const std::string& message, std::size_t lineNumber) {
    throw std::runtime_error("MpsReader Error (line " + std::to_string(lineNumber) +
                             "): " + message);
}

} // namespace

void MpsReader::reset() {
    current_section_ = MpsSection::NONE;
    objective_row_name_.clear();
    integer_marker_active_ = false;
    var_name_to_idx_.clear();
    constraint_name_to_idx_.clear();
    row_senses_.clear();
    selected_rhs_vector_.clear();
    selected_ranges_vector_.clear();
    row_rhs_.clear();
    row_range_.clear();
    row_has_range_.clear();
    free_row_names_.clear();
    var_lower_explicit_.clear();
    obj_term_map_.clear();
    constraint_term_maps_.clear();
    quad_term_map_.clear();
    objective_offset_ = 0.0;
    warnings_.clear();
}

void MpsReader::warn(const std::string& message) {
    warnings_.push_back(message);
}

model::Model MpsReader::read(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("MpsReader Error: Unable to open file " + filepath);
    }

    model::Model model;
    reset();

    // MPS has no sense keyword of its own; OBJSENSE is the extension that adds
    // one, and its absence means minimize.
    model.objective.sense = model::ObjectiveSense::Minimize;

    std::string line;
    std::size_t lineNumber = 0;

    while (std::getline(file, line)) {
        ++lineNumber;

        // Strip a trailing carriage return so files written on Windows parse
        // identically -- otherwise the last token on every line carries a \r,
        // and a bound type reads as "FR\r" and is rejected as unknown.
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (line.empty() || line[0] == '*') {
            continue;
        }

        std::stringstream ss(line);
        std::string token;
        if (!(ss >> token)) {
            continue;
        }

        // A section header starts in column 1; data records are indented.
        if (line[0] != ' ' && line[0] != '\t') {
            const std::string header = toUpper(token);

            if (header == "NAME") {
                current_section_ = MpsSection::NAME;
                if (ss >> token) {
                    model.name = token;
                }
            } else if (header == "OBJSENSE" || header == "OBJSENS") {
                current_section_ = MpsSection::OBJSENSE;
                // Both spellings occur: the sense on the header line, or on the
                // indented line that follows.
                if (ss >> token) {
                    const std::string sense = toUpper(token);
                    if (sense == "MAX" || sense == "MAXIMIZE") {
                        model.objective.sense = model::ObjectiveSense::Maximize;
                    } else if (sense == "MIN" || sense == "MINIMIZE") {
                        model.objective.sense = model::ObjectiveSense::Minimize;
                    } else {
                        fail("unknown OBJSENSE value '" + token + "'", lineNumber);
                    }
                }
            } else if (header == "ROWS") {
                current_section_ = MpsSection::ROWS;
            } else if (header == "COLUMNS") {
                current_section_ = MpsSection::COLUMNS;
            } else if (header == "RHS") {
                current_section_ = MpsSection::RHS;
            } else if (header == "RANGES") {
                current_section_ = MpsSection::RANGES;
            } else if (header == "BOUNDS") {
                current_section_ = MpsSection::BOUNDS;
            } else if (header == "QUADOBJ" || header == "QUADS") {
                current_section_ = MpsSection::QUADOBJ;
            } else if (header == "QMATRIX" || header == "QUADOBJ2") {
                current_section_ = MpsSection::QMATRIX;
            } else if (header == "ENDATA") {
                current_section_ = MpsSection::ENDATA;
                break;
            } else if (header == "SOS") {
                fail("SOS constraints are not representable in model::Model; "
                     "parsing this file would silently drop them", lineNumber);
            } else if (header == "INDICATORS") {
                fail("indicator constraints are not representable in model::Model; "
                     "parsing this file would silently drop them", lineNumber);
            } else if (header == "QCMATRIX" || header == "QCROWS") {
                fail("quadratic constraints are not representable in "
                     "model::Model (Constraint carries only linearTerms)", lineNumber);
            } else {
                // Never fall through to the previous section: that is exactly
                // how RANGES data used to be consumed as RHS data.
                fail("unrecognised section header '" + token + "'", lineNumber);
            }
            continue;
        }

        switch (current_section_) {
            case MpsSection::OBJSENSE: {
                const std::string sense = toUpper(token);
                if (sense == "MAX" || sense == "MAXIMIZE") {
                    model.objective.sense = model::ObjectiveSense::Maximize;
                } else if (sense == "MIN" || sense == "MINIMIZE") {
                    model.objective.sense = model::ObjectiveSense::Minimize;
                } else {
                    fail("unknown OBJSENSE value '" + token + "'", lineNumber);
                }
                break;
            }

            case MpsSection::ROWS: {
                const std::string sense = toUpper(token);
                std::string row_name;
                if (!(ss >> row_name)) {
                    fail("ROWS record missing a row name", lineNumber);
                }

                if (sense == "N") {
                    // The first free row is the objective; the rest are genuine
                    // free rows and the format expects them to be ignored.
                    if (objective_row_name_.empty()) {
                        objective_row_name_ = row_name;
                    } else {
                        free_row_names_.push_back(row_name);
                        warn("free row '" + row_name +
                             "' after the objective row was ignored");
                    }
                    break;
                }

                if (sense != "L" && sense != "G" && sense != "E") {
                    fail("unknown row sense '" + token + "' for row '" + row_name + "'",
                         lineNumber);
                }

                if (constraint_name_to_idx_.count(row_name) != 0) {
                    fail("duplicate row name '" + row_name + "'", lineNumber);
                }

                model::Constraint constraint;
                constraint.name = row_name;
                // Bounds are computed in finalizeRowBounds() once sense, RHS and
                // RANGES are all known.
                constraint.lowerBound = -kInfinity;
                constraint.upperBound = kInfinity;

                if (sense == "L") {
                    row_senses_[row_name] = RowSense::LESS_EQUAL;
                } else if (sense == "G") {
                    row_senses_[row_name] = RowSense::GREATER_EQUAL;
                } else {
                    row_senses_[row_name] = RowSense::EQUAL;
                }

                constraint_name_to_idx_[row_name] = model.constraints.size();
                model.constraints.push_back(std::move(constraint));
                constraint_term_maps_.emplace_back();
                row_rhs_.push_back(0.0);
                row_range_.push_back(0.0);
                row_has_range_.push_back(0);
                break;
            }

            case MpsSection::COLUMNS: {
                // Integer markers:
                //   MARKER_NAME  'MARKER'  'INTORG'
                // The name field is arbitrary, so detect on the keywords.
                std::istringstream markerFields(line);
                std::string markerName, markerToken, markerKind;
                if (markerFields >> markerName >> markerToken >> markerKind) {
                    markerToken = toUpper(unquote(markerToken));
                    markerKind = toUpper(unquote(markerKind));
                    if (markerToken == "MARKER" &&
                        (markerKind == "INTORG" || markerKind == "INTEND")) {
                        integer_marker_active_ = markerKind == "INTORG";
                        break;
                    }
                }

                const std::string var_name = token;
                std::string row_name;
                double value = 0.0;

                // Ensure the column exists even if the pair list is empty, so
                // its integrality is recorded by the marker that encloses it.
                get_or_create_variable(var_name, model);

                int pairs = 0;
                while (ss >> row_name) {
                    if (!(ss >> value)) {
                        fail("COLUMNS entry for '" + var_name + "' row '" + row_name +
                             "' has no coefficient", lineNumber);
                    }
                    add_coefficient(var_name, row_name, value, model);
                    ++pairs;
                }
                if (pairs == 0) {
                    warn("COLUMNS record for '" + var_name + "' carried no coefficients");
                }
                break;
            }

            case MpsSection::RHS: {
                // Keep all records belonging to the first RHS vector only,
                // including its objective constant. Other vectors are alternatives.
                // The set name in field 1 is OPTIONAL, and real files omit it.
                // Netlib's blend writes
                //     65   23.26   66   5.25
                // with no set name, so consuming field 1 unconditionally ate a
                // row name and the record then failed as "RHS entry for row
                // '5.25' has no value" -- the whole file was unreadable.
                //
                // Records are (row, value) PAIRS, so an odd field count means a
                // set name is present and an even count means it is not. Parity
                // decides it, not position, and the multi-vector selection below
                // only applies when a name was actually given.
                const std::vector<std::string> fields = splitFields(line);
                const bool named = (fields.size() % 2 == 1);
                const std::size_t start = named ? 1u : 0u;

                if (named) {
                    if (selected_rhs_vector_.empty()) selected_rhs_vector_ = fields[0];
                    if (fields[0] != selected_rhs_vector_) {
                        warn("RHS vector '" + fields[0] +
                             "' ignored; using first vector '" +
                             selected_rhs_vector_ + "'");
                        break;
                    }
                }

                std::string row_name;
                double value = 0.0;
                int pairs = 0;

                for (std::size_t k = start; k + 1 < fields.size(); k += 2) {
                    row_name = fields[k];
                    if (!parseNumber(fields[k + 1], value)) {
                        fail("RHS entry for row '" + row_name +
                             "' has a non-numeric value '" + fields[k + 1] + "'",
                             lineNumber);
                    }
                    ++pairs;

                    if (row_name == objective_row_name_) {
                        // MPS defines an objective-row RHS as the NEGATED
                        // objective constant. Dropping it (as this reader did)
                        // shifts every reported objective by that constant;
                        // adding it unnegated flips the shift's sign, which is
                        // just as wrong and harder to spot.
                        objective_offset_ = -value;
                        continue;
                    }

                    auto it = constraint_name_to_idx_.find(row_name);
                    if (it == constraint_name_to_idx_.end()) {
                        if (std::find(free_row_names_.begin(), free_row_names_.end(),
                                      row_name) == free_row_names_.end()) {
                            warn("RHS entry names unknown row '" + row_name +
                                 "'; ignored");
                        }
                        continue;
                    }
                    row_rhs_[it->second] = value;
                }

                if (pairs == 0) {
                    warn("RHS record carried no entries");
                }
                break;
            }

            case MpsSection::RANGES: {
                // Same optional set name as RHS, decided by the same parity.
                const std::vector<std::string> fields = splitFields(line);
                const bool named = (fields.size() % 2 == 1);
                const std::size_t start = named ? 1u : 0u;

                if (named) {
                    if (selected_ranges_vector_.empty()) selected_ranges_vector_ = fields[0];
                    if (fields[0] != selected_ranges_vector_) {
                        warn("RANGES vector '" + fields[0] +
                             "' ignored; using first vector '" +
                             selected_ranges_vector_ + "'");
                        break;
                    }
                }

                std::string row_name;
                double value = 0.0;

                for (std::size_t k = start; k + 1 < fields.size(); k += 2) {
                    row_name = fields[k];
                    if (!parseNumber(fields[k + 1], value)) {
                        fail("RANGES entry for row '" + row_name +
                             "' has a non-numeric value '" + fields[k + 1] + "'",
                             lineNumber);
                    }

                    if (row_name == objective_row_name_) {
                        warn("RANGES entry on the objective row was ignored");
                        continue;
                    }

                    auto it = constraint_name_to_idx_.find(row_name);
                    if (it == constraint_name_to_idx_.end()) {
                        warn("RANGES entry names unknown row '" + row_name + "'; ignored");
                        continue;
                    }
                    row_range_[it->second] = value;
                    row_has_range_[it->second] = 1;
                }
                break;
            }

            case MpsSection::BOUNDS: {
                const std::string bound_type = toUpper(unquote(token));
                std::string bound_label;
                std::string var_name;

                if (!(ss >> bound_label)) {
                    fail("BOUNDS record missing the bound-set name", lineNumber);
                }
                if (!(ss >> var_name)) {
                    fail("BOUNDS record missing a column name", lineNumber);
                }

                const int v_idx = get_or_create_variable(var_name, model);
                model::Variable& var = model.variables[static_cast<std::size_t>(v_idx)];

                // FR, MI, PL and BV take no value; the rest require one.
                const bool needsValue =
                    bound_type == "LO" || bound_type == "UP" || bound_type == "FX" ||
                    bound_type == "LI" || bound_type == "UI";
                double value = 0.0;
                if (needsValue && !(ss >> value)) {
                    fail("bound type '" + bound_type + "' on column '" + var_name +
                         "' requires a value", lineNumber);
                }

                if ((bound_type == "LI" || bound_type == "UI") &&
                    (!std::isfinite(value) || std::trunc(value) != value)) {
                    fail("bound type '" + bound_type + "' on column '" + var_name +
                         "' requires an integral value", lineNumber);
                }

                if (bound_type == "LO") {
                    var.lowerBound = value;
                    var_lower_explicit_[static_cast<std::size_t>(v_idx)] = 1;
                } else if (bound_type == "UP") {
                    var.upperBound = value;
                    // A negative UP on a column whose lower bound is still the
                    // implicit 0 would otherwise produce lb > ub and a model
                    // that fails validation. Every mainstream reader opens the
                    // lower bound instead; doing anything else here turns a
                    // large family of real instances into parse failures.
                    if (value < 0.0 &&
                        !var_lower_explicit_[static_cast<std::size_t>(v_idx)] &&
                        var.lowerBound == 0.0) {
                        var.lowerBound = -kInfinity;
                        warn("negative UP bound on '" + var_name +
                             "' opened its lower bound to -infinity");
                    }
                } else if (bound_type == "FX") {
                    var.lowerBound = value;
                    var.upperBound = value;
                    var_lower_explicit_[static_cast<std::size_t>(v_idx)] = 1;
                } else if (bound_type == "FR") {
                    var.lowerBound = -kInfinity;
                    var.upperBound = kInfinity;
                    var_lower_explicit_[static_cast<std::size_t>(v_idx)] = 1;
                } else if (bound_type == "MI") {
                    var.lowerBound = -kInfinity;
                    var_lower_explicit_[static_cast<std::size_t>(v_idx)] = 1;
                } else if (bound_type == "PL") {
                    var.upperBound = kInfinity;
                } else if (bound_type == "BV") {
                    var.lowerBound = 0.0;
                    var.upperBound = 1.0;
                    var.type = model::VariableType::Binary;
                    var_lower_explicit_[static_cast<std::size_t>(v_idx)] = 1;
                } else if (bound_type == "LI") {
                    var.lowerBound = value;
                    var_lower_explicit_[static_cast<std::size_t>(v_idx)] = 1;
                    if (var.type == model::VariableType::Continuous) {
                        var.type = model::VariableType::Integer;
                    }
                } else if (bound_type == "UI") {
                    var.upperBound = value;
                    if (var.type == model::VariableType::Continuous) {
                        var.type = model::VariableType::Integer;
                    }
                } else {
                    fail("unknown bound type '" + token + "' on column '" + var_name + "'",
                         lineNumber);
                }
                break;
            }

            case MpsSection::QUADOBJ:
            case MpsSection::QMATRIX: {
                // QUADOBJ lists the lower triangle of Q; QMATRIX lists all of
                // it. Both describe the same objective, 0.5 * x' Q x.
                const std::string col1 = token;
                std::string col2;
                double value = 0.0;

                while (ss >> col2) {
                    if (!(ss >> value)) {
                        fail("quadratic entry for '" + col1 + "', '" + col2 +
                             "' has no value", lineNumber);
                    }
                    add_quadratic(col1, col2, value,
                                  current_section_ == MpsSection::QUADOBJ, model);
                }
                break;
            }

            case MpsSection::NAME:
            case MpsSection::NONE:
            case MpsSection::ENDATA:
                break;
        }
    }

    // Flatten the accumulators. Terms were summed by index along the way, so a
    // file that writes the same (row, column) pair twice produces one entry
    // rather than two structural nonzeros.
    for (const auto& entry : obj_term_map_) {
        model.objective.linearTerms.push_back(
            model::LinearTerm{entry.first, entry.second});
    }
    for (std::size_t i = 0; i < model.constraints.size(); ++i) {
        for (const auto& entry : constraint_term_maps_[i]) {
            model.constraints[i].linearTerms.push_back(
                model::LinearTerm{entry.first, entry.second});
        }
    }
    for (const auto& entry : quad_term_map_) {
        if (entry.second == 0.0) {
            continue;
        }
        model.objective.quadraticTerms.push_back(model::QuadraticTerm{
            entry.first.first, entry.first.second, entry.second});
    }

    model.objective.offset = objective_offset_;
    finalizeRowBounds(model);

    return model;
}

void MpsReader::finalizeRowBounds(model::Model& model) {
    for (std::size_t i = 0; i < model.constraints.size(); ++i) {
        model::Constraint& constraint = model.constraints[i];
        const RowSense sense = row_senses_[constraint.name];
        const double rhs = row_rhs_[i];

        if (!row_has_range_[i]) {
            switch (sense) {
                case RowSense::LESS_EQUAL:
                    constraint.lowerBound = -kInfinity;
                    constraint.upperBound = rhs;
                    break;
                case RowSense::GREATER_EQUAL:
                    constraint.lowerBound = rhs;
                    constraint.upperBound = kInfinity;
                    break;
                case RowSense::EQUAL:
                    constraint.lowerBound = rhs;
                    constraint.upperBound = rhs;
                    break;
            }
            continue;
        }

        // RANGES turns a one-sided row into a two-sided one. The width is
        // |R| in every case; only an equality row takes its side from the
        // SIGN of R, which is the part readers most often get wrong.
        const double range = row_range_[i];
        const double width = std::abs(range);

        switch (sense) {
            case RowSense::LESS_EQUAL:
                constraint.lowerBound = rhs - width;
                constraint.upperBound = rhs;
                break;
            case RowSense::GREATER_EQUAL:
                constraint.lowerBound = rhs;
                constraint.upperBound = rhs + width;
                break;
            case RowSense::EQUAL:
                if (range >= 0.0) {
                    constraint.lowerBound = rhs;
                    constraint.upperBound = rhs + range;
                } else {
                    constraint.lowerBound = rhs + range;
                    constraint.upperBound = rhs;
                }
                break;
        }
    }
}

int MpsReader::get_or_create_variable(const std::string& var_name, model::Model& model) {
    auto it = var_name_to_idx_.find(var_name);
    if (it != var_name_to_idx_.end()) {
        return static_cast<int>(it->second);
    }

    const int new_idx = static_cast<int>(model.variables.size());
    model::Variable var;
    var.name = var_name;
    // An integer column's default upper bound is +infinity, matching every
    // mainstream solver. The older [0, 1] default silently turns general
    // integers into binaries.
    var.type = integer_marker_active_ ? model::VariableType::Integer
                                      : model::VariableType::Continuous;
    var.lowerBound = 0.0;
    var.upperBound = kInfinity;

    model.variables.push_back(std::move(var));
    var_name_to_idx_[var_name] = static_cast<std::size_t>(new_idx);
    var_lower_explicit_.push_back(0);
    return new_idx;
}

void MpsReader::add_coefficient(const std::string& var_name, const std::string& row_name,
                                double value, model::Model& model) {
    const int v_idx = get_or_create_variable(var_name, model);

    if (row_name == objective_row_name_) {
        obj_term_map_[v_idx] += value;
        return;
    }

    auto it = constraint_name_to_idx_.find(row_name);
    if (it != constraint_name_to_idx_.end()) {
        constraint_term_maps_[it->second][v_idx] += value;
        return;
    }

    // A coefficient on a free row is discarded by design. A coefficient on a
    // row that was never declared is a malformed file, but it cannot change the
    // model that does exist, so it is reported rather than fatal.
    if (std::find(free_row_names_.begin(), free_row_names_.end(), row_name) ==
        free_row_names_.end()) {
        warn("COLUMNS entry for '" + var_name + "' names unknown row '" + row_name +
             "'; ignored");
    }
}

void MpsReader::add_quadratic(const std::string& col1, const std::string& col2,
                              double value, bool mirror, model::Model& model) {
    const int i = get_or_create_variable(col1, model);
    const int j = get_or_create_variable(col2, model);

    // The file describes 0.5 * x' Q x. model::Model's QuadraticTerm carries the
    // DIRECT coefficient of x_i*x_j with no implicit 0.5 (see model.h), so:
    //
    //   diagonal      0.5 * Q_ii * x_i^2         -> q_ii = Q_ii / 2
    //   off-diagonal  0.5 * (Q_ij + Q_ji) x_i x_j
    //
    // QUADOBJ lists an off-diagonal once and implies its mirror, so that entry
    // contributes the full Q_ij. QMATRIX lists both, so each contributes half
    // and the two sum to the same total. Halving both -- or neither -- gets one
    // of the two formats wrong by a factor of two, which is invisible in the
    // status and wrong in the objective.
    const int lo = std::min(i, j);
    const int hi = std::max(i, j);

    if (i == j) {
        quad_term_map_[{lo, hi}] += value * 0.5;
    } else {
        quad_term_map_[{lo, hi}] += mirror ? value : value * 0.5;
    }
}

} // namespace mps
