#include "nlp/solver.h"
#include "nlp/derivative_check.h"
#include <future>
#include <cmath>
#include <iostream>
#include <sstream>
#include <stdexcept>
using namespace nlp;
void check(bool p,const std::string& what) {if(!p) throw std::runtime_error(what);}
void near(double a,double b,double tol=1e-5) {check(std::isfinite(a)&&std::abs(a-b)<=tol,"expected "+std::to_string(b)+", got "+std::to_string(a));}
Result solve(const Model& p,std::vector<double> x,Options o={}) {
    auto r=Solver().solve(p,x,o);
    std::cout<<toString(r.status)<<" f="<<r.objective<<" pr="<<r.primalResidual<<" du="<<r.dualResidual<<" comp="<<r.complementarity<<" it="<<r.iteration<<" "<<r.message<<std::endl;
    check(r.status==Status::FirstOrderStationary,"solve failed");
    check(r.feasible&&r.hasPrimal,"missing solution");
    check(r.primalResidual<=o.tolerance&&r.dualResidual<=o.tolerance&&r.complementarity<=o.tolerance,"KKT gate");
    return r;
}
class BadCallback : public Problem {
public:
    std::vector<Bounds> bounds{{}};
    std::vector<Bounds> empty;
    int mode = 0;
    const std::vector<Bounds>& variableBounds() const override { return bounds; }
    const std::vector<Bounds>& constraintBounds() const override { return empty; }
    Evaluation evaluate(const std::vector<double>& x) const override {
        Evaluation e; e.objective = x[0]*x[0]; e.gradient = {0};
        e.jacobian = qp::SparseMatrix::fromTriplets(0,1,{},{},{});
        if (mode == 1) e.gradient.clear();
        if (mode == 2) e.objective = std::numeric_limits<double>::quiet_NaN();
        if (mode == 3) throw std::runtime_error("callback failed");
        return e;
    }
};
int main() {
    try {
        auto x=variable(0),y=variable(1);
        // AD against central differences, including shared subexpressions.
        auto shared=sin(x*y)+exp(x/3.0);
        Model ad({{},{}},square(shared)+log(y)+sqrt(y)+cos(x)-x/y,{shared,square(x)},{ {},{} });
        check(checkDerivatives(ad,{0.4,1.3}).passed,"derivative checker");
        BadCallback bad;
        check(!checkDerivatives(bad,{1}).passed,"incorrect derivative accepted");
        bad.mode=1;check(Solver().solve(bad,{1}).status==Status::InvalidProblem,"bad callback shape");
        bad.mode=2;check(Solver().solve(bad,{1}).status==Status::EvaluationFailure,"NaN callback accepted");
        bad.mode=3;check(Solver().solve(bad,{1}).status==Status::NumericalFailure,"callback exception escaped");
        Model repeated({{}},x,{x,x},{{},{}});
        auto repeatedEvaluation=repeated.evaluate({2});
        std::vector<double> repeatedProduct;
        repeatedEvaluation.jacobian.multiply({1},repeatedProduct);
        near(repeatedEvaluation.gradient[0],1);
        near(repeatedProduct[0],1);near(repeatedProduct[1],1);
        // Compilation, evaluation AND destruction must tolerate a deep DAG.
        {
            Expression chain=x;
            for(int k=0;k<20000;++k) chain=chain+1;
            Model deep({{}},chain);
            auto value=deep.evaluate({2});near(value.objective,20002);near(value.gradient[0],1);
        }
        auto e=ad.evaluate({0.4,1.3});
        for(int j=0;j<2;++j) {
            std::vector<double> a{0.4,1.3},b=a;a[j]+=1e-6;b[j]-=1e-6;
            auto ea=ad.evaluate(a),eb=ad.evaluate(b);
            near(e.gradient[j],(ea.objective-eb.objective)/2e-6,1e-7);
            std::vector<double> unit(2,0),col;unit[j]=1;e.jacobian.multiply(unit,col);
            for(int i=0;i<2;++i) near(col[i],(ea.constraints[i]-eb.constraints[i])/2e-6,1e-7);
        }
        near(solve(Model({{}},square(x-3)),{0}).primal[0],3);
        auto eq=solve(Model({{},{}},square(x)+square(y),{x+y},{{2,2}}),{0,0});
        near(eq.primal[0],1);near(eq.primal[1],1);near(eq.constraintMultipliers[0],-2);
        auto lower=solve(Model({{}},square(x),{x},{{2,infinity}}),{0});
        near(lower.primal[0],2);near(lower.constraintMultipliers[0],-4);
        auto upper=solve(Model({{}},square(x-3),{x},{{-infinity,1}}),{0});near(upper.constraintMultipliers[0],4);
        auto ranged=solve(Model({{}},square(x-3),{x},{{1,2}}),{0});
        near(ranged.primal[0],2);near(ranged.constraintMultipliers[0],2);
        auto inactive=solve(Model({{}},square(x-1.5),{x},{{1,2}}),{1});
        near(inactive.primal[0],1.5);near(inactive.constraintMultipliers[0],0);
        auto bound=solve(Model({{0,1}},square(x-3)),{10});near(bound.boundMultipliers[0],4);
        auto fixed=solve(Model({{2,2}},square(x)),{-1});near(fixed.primal[0],2);
        auto rosen=solve(Model({{},{}},100*square(y-square(x))+square(1-x)),{-1.2,1});near(rosen.objective,0,1e-10);
        auto circle=solve(Model({{},{}},-x-y,{square(x)+square(y)},{{-infinity,1}}),{0.2,0.1});near(circle.objective,-std::sqrt(2.0));
        // Duplicate equalities: singular equality Jacobian must not require inversion.
        solve(Model({{},{}},square(x)+square(y),{x+y,2*(x+y)},{{2,2},{4,4}}),{0,0});
        auto restoration=solve(Model({{-2,2}},square(x),{square(x)},{{1,1}}),{0.1});
        check(restoration.elasticSubproblems>0,"restoration was not exercised");
        near(restoration.primal[0],1);
        // Nonconvex objective: local stationary solution, never global Optimal.
        auto nc=solve(Model({{-2,2}},square(square(x)-1)),{0.3});near(nc.primal[0],1);
        // Hock-Schittkowski 71, nonlinear equality and inequality, bounded.
        auto z=variable(2),w=variable(3);
        auto hs=solve(Model({{1,5},{1,5},{1,5},{1,5}},x*w*(x+y+z)+z,
            {x*y*z*w,square(x)+square(y)+square(z)+square(w)},{{25,infinity},{40,40}}),{1,5,5,1});
        near(hs.objective,17.0140173,1e-5);
        // Domain backtracking must retain a valid point and converge.
        auto logProblem=solve(Model({{}},x-log(x)),{4});
        near(logProblem.primal[0],1);check(logProblem.rejectedTrials>0,"domain backtracking not exercised");
        auto invalid=Solver().solve(Model({{2,1}},square(x)),{0});check(invalid.status==Status::InvalidProblem,"bad bounds");
        auto saddle=Solver().solve(Model({{}},-square(x)),{0});
        check(saddle.status==Status::FirstOrderStationary,"first-order status contract");
        auto overflow=Solver().solve(Model({{}},exp(x)),{1000});
        check(overflow.status==Status::EvaluationFailure,"overflow falsely accepted");
        auto domain=Solver().solve(Model({{}},log(x)),{-1});check(domain.status==Status::EvaluationFailure,"domain status");
        Options limit;limit.iterationLimit=0;
        check(Solver().solve(Model({{}},square(x-1)),{0},limit).status==Status::IterationLimit,"limit status");
        Options stop;stop.callback=[](const Iteration&){return false;};
        check(Solver().solve(Model({{}},square(x-1)),{0},stop).status==Status::UserStopped,"callback");
        auto impossible=Solver().solve(Model({{}},square(x),{square(x)},{{-infinity,-1}}),{0});
        check(impossible.status!=Status::FirstOrderStationary&&!impossible.feasible,"false infeasibility success");
        auto constant=solve(Model({},Expression(3)),{});near(constant.objective,3);
        Options scaled;scaled.variableScale={1e3,1e-3};
        solve(Model({{},{}},square(x/1000-1)+square(1000*y-1),{x/1000+1000*y},{{2,2}}),{0,0},scaled);
        Options diagonal;diagonal.denseBfgsLimit=0;
        solve(Model({{},{}},square(x-1)+square(y-2)),{0,0},diagonal);
        std::istringstream input("nlp 1 variables 1 -inf inf 0 nodes 4 var 0 const 3 sub 0 1 square 2 objective 3 constraints 0");
        auto parsed=read(input);near(solve(parsed.model,parsed.initial).primal[0],3);
        Options invalidOptions;invalidOptions.variableScale={0};
        check(Solver().solve(Model({{}},square(x)),{1},invalidOptions).status==Status::InvalidProblem,"invalid scale");
        invalidOptions.variableScale.clear();invalidOptions.tolerance=std::numeric_limits<double>::quiet_NaN();
        check(Solver().solve(Model({{}},square(x)),{1},invalidOptions).status==Status::InvalidProblem,"NaN tolerance");
        Options tinyTime;tinyTime.timeLimitSeconds=1e-12;
        check(Solver().solve(Model({{}},square(x)),{1},tinyTime).status==Status::TimeLimit,"time limit");
        Options tinyQp;tinyQp.qpIterationLimit=1;
        check(Solver().solve(Model({{}},square(x),{x},{{2,infinity}}),{0},tinyQp).status==Status::SubproblemFailure,"QP budget falsely accepted");
        // Sparse dimension exceeds dense BFGS threshold. Shared model is reentrant.
        Expression bigObjective(0);std::vector<Bounds> many(300);
        for(int j=0;j<300;++j) bigObjective=bigObjective+square(variable(j)-1);
        Model sparse(many,bigObjective);
        auto future=std::async(std::launch::async,[&]{return Solver().solve(sparse,std::vector<double>(300,0));});
        auto sparseResult=solve(sparse,std::vector<double>(300,2));
        auto concurrent=future.get();check(concurrent.status==Status::FirstOrderStationary,"concurrent solve");
        near(sparseResult.primal[299],1);
        std::cout<<"NLP tests passed\n";
    } catch(const std::exception& e) {std::cerr<<e.what()<<"\n";return 1;}
}
