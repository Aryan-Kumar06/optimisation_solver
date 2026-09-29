#pragma once
#include "qp/qp_model.h"
#include <limits>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

namespace nlp {
constexpr double infinity = std::numeric_limits<double>::infinity();
struct Bounds { double lower = -infinity, upper = infinity; };
struct Evaluation {
    double objective = 0;
    std::vector<double> gradient, constraints;
    qp::SparseMatrix jacobian;
};
// Callback implementations must supply finite values and first derivatives in
// original units. Throw std::domain_error for recoverable evaluation failures.
class Problem {
public:
    virtual ~Problem() = default;
    virtual const std::vector<Bounds>& variableBounds() const = 0;
    virtual const std::vector<Bounds>& constraintBounds() const = 0;
    virtual Evaluation evaluate(const std::vector<double>& x) const = 0;
};

enum class Op { Constant, Variable, Add, Subtract, Multiply, Divide, Negate,
                Exp, Log, Sin, Cos, Sqrt, Square };
struct ExpressionBuilder;
class Expression {
public:
    Expression(double value = 0);
private:
    friend class Model;
    friend struct ExpressionBuilder;
    struct Node {
        Op op;
        double value;
        int variable;
        std::shared_ptr<const Node> left, right;
        mutable const Node* disposalNext = nullptr;
    };
    explicit Expression(std::shared_ptr<const Node> node) : node_(std::move(node)) {}
    std::shared_ptr<const Node> node_;
};
Expression variable(int index);
Expression operator+(Expression a, Expression b);
Expression operator-(Expression a, Expression b);
Expression operator*(Expression a, Expression b);
Expression operator/(Expression a, Expression b);
Expression operator-(Expression a);
Expression exp(Expression a);
Expression log(Expression a);
Expression sin(Expression a);
Expression cos(Expression a);
Expression sqrt(Expression a);
Expression square(Expression a);

// Immutable compiled expression DAG. Evaluation scratch is per call, making
// concurrent solves safe. No global tape, variable registry, or mutable cache.
class Model final : public Problem {
public:
    Model(std::vector<Bounds> variables, Expression objective,
          std::vector<Expression> constraints = {}, std::vector<Bounds> bounds = {});
    const std::vector<Bounds>& variableBounds() const override { return variables_; }
    const std::vector<Bounds>& constraintBounds() const override { return bounds_; }
    Evaluation evaluate(const std::vector<double>& x) const override;
private:
    struct Instruction { Op op; double value; int variable, left, right; };
    std::vector<Bounds> variables_, bounds_;
    std::vector<Instruction> tape_;
    std::vector<int> roots_;
    std::vector<std::vector<int>> reachable_;
};
// Safe, versioned text format (see README); no evaluation of host-language code.
struct Input { Model model; std::vector<double> initial; };
Input read(std::istream& stream);
} // namespace nlp
