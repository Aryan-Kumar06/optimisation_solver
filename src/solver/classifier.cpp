#include "solver/classifier.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <unordered_map>
#include <vector>

namespace solver {
namespace {

constexpr double kEps = 1e-9;

bool isOne(double value) noexcept {
    return std::abs(value - 1.0) <= kEps;
}

bool isMinusOne(double value) noexcept {
    return std::abs(value + 1.0) <= kEps;
}

// A column-major view of the constraint matrix. model::Model stores rows, but
// network structure and symmetry are both column properties.
struct ColumnView {
    // column -> (row, coefficient), sorted by row
    std::vector<std::vector<std::pair<std::size_t, double>>> columns;
};

ColumnView buildColumnView(const model::Model& model) {
    ColumnView view;
    view.columns.resize(model.variables.size());
    for (std::size_t row = 0; row < model.constraints.size(); ++row) {
        for (const auto& term : model.constraints[row].linearTerms) {
            if (term.variableIndex < 0 ||
                static_cast<std::size_t>(term.variableIndex) >= view.columns.size()) {
                continue;
            }
            view.columns[static_cast<std::size_t>(term.variableIndex)]
                .emplace_back(row, term.value);
        }
    }
    for (auto& column : view.columns) {
        std::sort(column.begin(), column.end());
    }
    return view;
}

}  // namespace

const char* toString(ProblemClass value) noexcept {
    switch (value) {
        case ProblemClass::NLP:  return "NLP";
        case ProblemClass::LP:   return "LP";
        case ProblemClass::MILP: return "MILP";
        case ProblemClass::QP:   return "QP";
        case ProblemClass::MIQP: return "MIQP";
        case ProblemClass::QCQP: return "QCQP";
    }
    return "unknown";
}

Classification classify(const model::Model& model) {
    Classification result;
    StructureHints& hints = result.hints;

    hints.numRows = model.constraints.size();
    hints.numColumns = model.variables.size();

    for (const auto& variable : model.variables) {
        switch (variable.type) {
            case model::VariableType::Binary:     ++hints.numBinary; break;
            case model::VariableType::Integer:    ++hints.numInteger; break;
            case model::VariableType::Continuous: ++hints.numContinuous; break;
        }
    }

    // A variable declared Integer whose bounds pin it to [0,1] is a binary in
    // everything but name, and the MPS reader has no obligation to say so.
    for (const auto& variable : model.variables) {
        if (variable.type == model::VariableType::Integer &&
            variable.lowerBound >= -kEps && variable.upperBound <= 1.0 + kEps) {
            --hints.numInteger;
            ++hints.numBinary;
        }
    }

    const bool hasIntegrality = (hints.numBinary + hints.numInteger) > 0;
    const bool hasQuadratic = !model.objective.quadraticTerms.empty();

    if (hasQuadratic) {
        result.problemClass = hasIntegrality ? ProblemClass::MIQP : ProblemClass::QP;
    } else {
        result.problemClass = hasIntegrality ? ProblemClass::MILP : ProblemClass::LP;
    }

    double smallest = std::numeric_limits<double>::infinity();
    double largest = 0.0;
    for (const auto& constraint : model.constraints) {
        for (const auto& term : constraint.linearTerms) {
            const double magnitude = std::abs(term.value);
            if (magnitude <= kEps) {
                continue;
            }
            ++hints.numNonzeros;
            smallest = std::min(smallest, magnitude);
            largest = std::max(largest, magnitude);
        }
    }
    hints.coefRangeRatio =
        (hints.numNonzeros > 0 && smallest > 0.0) ? largest / smallest : 0.0;

    if (model.variables.empty() || model.constraints.empty()) {
        return result;
    }

    const ColumnView view = buildColumnView(model);

    // Network structure: every non-empty column is exactly one +1 and one -1.
    hints.hasNetworkStructure = true;
    for (const auto& column : view.columns) {
        if (column.empty()) {
            continue;
        }
        if (column.size() != 2) {
            hints.hasNetworkStructure = false;
            break;
        }
        const bool pair = (isOne(column[0].second) && isMinusOne(column[1].second)) ||
                          (isMinusOne(column[0].second) && isOne(column[1].second));
        if (!pair) {
            hints.hasNetworkStructure = false;
            break;
        }
    }

    for (const auto& constraint : model.constraints) {
        if (constraint.linearTerms.empty()) {
            continue;
        }

        // Big-M: a row mixing binaries and non-binaries where some binary's
        // coefficient dwarfs the rest of the row.
        double largestBinary = 0.0;
        double largestOther = 0.0;
        bool sawBinary = false;
        bool sawOther = false;
        bool allBinary = true;
        bool allUnitCoefficients = true;

        for (const auto& term : constraint.linearTerms) {
            const auto index = static_cast<std::size_t>(term.variableIndex);
            if (index >= model.variables.size()) {
                continue;
            }
            const model::Variable& variable = model.variables[index];
            const bool binary =
                variable.type == model::VariableType::Binary ||
                (variable.type == model::VariableType::Integer &&
                 variable.lowerBound >= -kEps && variable.upperBound <= 1.0 + kEps);

            const double magnitude = std::abs(term.value);
            if (binary) {
                sawBinary = true;
                largestBinary = std::max(largestBinary, magnitude);
            } else {
                sawOther = true;
                allBinary = false;
                largestOther = std::max(largestOther, magnitude);
            }
            if (!isOne(term.value)) {
                allUnitCoefficients = false;
            }
        }

        if (sawBinary && sawOther && largestOther > kEps &&
            largestBinary >= 100.0 * largestOther) {
            hints.hasBigM = true;
            hints.maxBigM = std::max(hints.maxBigM, largestBinary);
        }

        // Set partitioning: all binary, unit coefficients, equality with rhs 1.
        if (allBinary && allUnitCoefficients &&
            std::abs(constraint.lowerBound - constraint.upperBound) <= kEps &&
            isOne(constraint.upperBound)) {
            hints.hasSetPartitioning = true;
        }
    }

    // Symmetry: group structurally identical columns. Two columns are
    // interchangeable only if their coefficients, bounds, type and objective
    // coefficient all agree.
    std::vector<double> objectiveCoefficient(model.variables.size(), 0.0);
    for (const auto& term : model.objective.linearTerms) {
        const auto index = static_cast<std::size_t>(term.variableIndex);
        if (index < objectiveCoefficient.size()) {
            objectiveCoefficient[index] += term.value;
        }
    }

    std::map<std::string, int> signatureCounts;
    for (std::size_t column = 0; column < view.columns.size(); ++column) {
        if (view.columns[column].empty()) {
            continue;
        }
        std::string signature;
        signature.reserve(view.columns[column].size() * 24);
        for (const auto& entry : view.columns[column]) {
            signature += std::to_string(entry.first);
            signature += ':';
            signature += std::to_string(entry.second);
            signature += ';';
        }
        const model::Variable& variable = model.variables[column];
        signature += '|';
        signature += std::to_string(static_cast<int>(variable.type));
        signature += ':';
        signature += std::to_string(variable.lowerBound);
        signature += ':';
        signature += std::to_string(variable.upperBound);
        signature += ':';
        signature += std::to_string(objectiveCoefficient[column]);
        ++signatureCounts[signature];
    }
    for (const auto& entry : signatureCounts) {
        if (entry.second >= 2) {
            ++hints.symmetricGroups;
        }
    }

    return result;
}

}  // namespace solver
