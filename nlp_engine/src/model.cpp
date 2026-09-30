#include "nlp/model.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <stdexcept>
#include <unordered_map>

namespace nlp {
struct ExpressionBuilder {
// Shared ownership normally destroys a deep expression chain recursively.
// Drain last-owned nodes iteratively, without allocating during destruction.
static void destroy(const Expression::Node* node) noexcept {
    thread_local const Expression::Node* pending = nullptr;
    thread_local bool draining = false;
    node->disposalNext = pending;
    pending = node;
    if (draining) return;
    draining = true;
    while (pending) {
        const auto* next = pending;
        pending = next->disposalNext;
        delete next; // Child deleters append to pending while draining is true.
    }
    draining = false;
}
static std::shared_ptr<const Expression::Node> own(Expression::Node node) {
    return {new Expression::Node(std::move(node)), destroy};
}
static Expression make(Op op, Expression a) {
    return Expression(own(Expression::Node{op,0,-1,a.node_,{}}));
}
static Expression make(Op op, Expression a, Expression b) {
    return Expression(own(Expression::Node{op,0,-1,a.node_,b.node_}));
}
static Expression var(int i) {
    return Expression(own(Expression::Node{Op::Variable,0,i,{},{}}));
}
};
namespace {
void finite(double v) {
    if (!std::isfinite(v)) throw std::domain_error("non-finite expression value or derivative");
}
}
Expression::Expression(double v) : node_(ExpressionBuilder::own(Node{Op::Constant,v,-1,{},{}})) {
    if (!std::isfinite(v)) throw std::invalid_argument("expression constant must be finite");
}
Expression variable(int i) {
    if (i < 0) throw std::invalid_argument("negative variable index");
    return ExpressionBuilder::var(i);
}
Expression operator+(Expression a, Expression b) { return ExpressionBuilder::make(Op::Add,a,b); }
Expression operator-(Expression a, Expression b) { return ExpressionBuilder::make(Op::Subtract,a,b); }
Expression operator*(Expression a, Expression b) { return ExpressionBuilder::make(Op::Multiply,a,b); }
Expression operator/(Expression a, Expression b) { return ExpressionBuilder::make(Op::Divide,a,b); }
Expression operator-(Expression a) { return ExpressionBuilder::make(Op::Negate,a); }
Expression exp(Expression a) { return ExpressionBuilder::make(Op::Exp,a); }
Expression log(Expression a) { return ExpressionBuilder::make(Op::Log,a); }
Expression sin(Expression a) { return ExpressionBuilder::make(Op::Sin,a); }
Expression cos(Expression a) { return ExpressionBuilder::make(Op::Cos,a); }
Expression sqrt(Expression a) { return ExpressionBuilder::make(Op::Sqrt,a); }
Expression square(Expression a) { return ExpressionBuilder::make(Op::Square,a); }

Model::Model(std::vector<Bounds> vars, Expression objective,
             std::vector<Expression> constraints, std::vector<Bounds> bounds)
    : variables_(std::move(vars)), bounds_(std::move(bounds)) {
    if (variables_.size() > 1000000 || constraints.size() > 1000000)
        throw std::invalid_argument("model exceeds supported index/resource limits");
    if (constraints.size() != bounds_.size()) throw std::invalid_argument("constraint bounds size mismatch");
    constraints.insert(constraints.begin(), objective);
    std::unordered_map<const Expression::Node*, int> ids;
    // Iterative postorder avoids recursion on long expression chains.
    for (const auto& expression : constraints) {
        std::vector<std::pair<std::shared_ptr<const Expression::Node>, bool>> stack{{expression.node_,false}};
        while (!stack.empty()) {
            auto entry = stack.back(); stack.pop_back();
            const auto& p = entry.first;
            if (!p) throw std::invalid_argument("null expression node");
            if (ids.count(p.get())) continue;
            if (!entry.second) {
                stack.push_back({p,true});
                if (p->right) stack.push_back({p->right,false});
                if (p->left) stack.push_back({p->left,false});
                continue;
            }
            if (p->op == Op::Variable && (p->variable < 0 || static_cast<size_t>(p->variable) >= variables_.size()))
                throw std::invalid_argument("variable index outside model");
            if (tape_.size() >= static_cast<size_t>(std::numeric_limits<int>::max()))
                throw std::invalid_argument("expression tape too large");
            int id = static_cast<int>(tape_.size());
            tape_.push_back({p->op,p->value,p->variable,p->left ? ids.at(p->left.get()) : -1,p->right ? ids.at(p->right.get()) : -1});
            ids.emplace(p.get(),id);
        }
        roots_.push_back(ids.at(expression.node_.get()));
    }
    std::vector<int> seen(tape_.size(),-1);
    int generation = 0;
    for (int root : roots_) {
        ++generation;
        std::vector<int> visit{root}, active;
        while (!visit.empty()) {
            int k = visit.back(); visit.pop_back();
            if (seen[k] == generation) continue;
            seen[k] = generation; active.push_back(k);
            if (tape_[k].left >= 0) visit.push_back(tape_[k].left);
            if (tape_[k].right >= 0) visit.push_back(tape_[k].right);
        }
        std::sort(active.begin(),active.end(),std::greater<int>());
        reachable_.push_back(std::move(active));
    }
}
Evaluation Model::evaluate(const std::vector<double>& x) const {
    if (x.size() != variables_.size()) throw std::invalid_argument("evaluation dimension mismatch");
    for (double v : x) finite(v);
    std::vector<double> values(tape_.size()), dl(tape_.size()), dr(tape_.size()), adj(tape_.size());
    for (size_t k=0;k<tape_.size();++k) {
        auto t=tape_[k]; double a=t.left<0 ? 0 : values[t.left], b=t.right<0 ? 0 : values[t.right];
        switch(t.op) {
        case Op::Constant: values[k]=t.value; break;
        case Op::Variable: values[k]=x[t.variable]; break;
        case Op::Add: values[k]=a+b; dl[k]=dr[k]=1; break;
        case Op::Subtract: values[k]=a-b; dl[k]=1; dr[k]=-1; break;
        case Op::Multiply: values[k]=a*b; dl[k]=b; dr[k]=a; break;
        case Op::Divide: values[k]=a/b; dl[k]=1/b; dr[k]=-values[k]/b; break;
        case Op::Negate: values[k]=-a; dl[k]=-1; break;
        case Op::Exp: values[k]=std::exp(a); dl[k]=values[k]; break;
        case Op::Log: values[k]=std::log(a); dl[k]=1/a; break;
        case Op::Sin: values[k]=std::sin(a); dl[k]=std::cos(a); break;
        case Op::Cos: values[k]=std::cos(a); dl[k]=-std::sin(a); break;
        case Op::Sqrt: values[k]=std::sqrt(a); dl[k]=0.5/values[k]; break;
        case Op::Square: values[k]=a*a; dl[k]=2*a; break;
        default: throw std::invalid_argument("unknown expression operation");
        }
        finite(values[k]); finite(dl[k]); finite(dr[k]);
    }
    Evaluation out; out.objective=values[roots_[0]]; out.gradient.assign(x.size(),0);
    std::vector<double> rows,cols,vals;
    for (size_t r=0;r<roots_.size();++r) {
        for (int k : reachable_[r]) adj[k]=0;
        adj[roots_[r]]=1;
        // Accumulate duplicate variable nodes before constructing CSR.
        for (int k : reachable_[r]) {
            auto t=tape_[k]; finite(adj[k]);
            if (t.op==Op::Variable) {
                if (r==0) { out.gradient[t.variable]+=adj[k]; finite(out.gradient[t.variable]); }
                else if (adj[k]!=0) { rows.push_back(static_cast<double>(r-1)); cols.push_back(t.variable); vals.push_back(adj[k]); }
            }
            if(t.left>=0) adj[t.left]+=adj[k]*dl[k];
            if(t.right>=0) adj[t.right]+=adj[k]*dr[k];
        }
        if(r) out.constraints.push_back(values[roots_[r]]);
    }
    out.jacobian=qp::SparseMatrix::fromTriplets(static_cast<int>(bounds_.size()),static_cast<int>(x.size()),rows,cols,vals);
    for (double v : out.jacobian.csrValues()) finite(v);
    return out;
}
} // namespace nlp
