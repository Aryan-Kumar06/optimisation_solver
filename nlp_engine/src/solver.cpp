#include "nlp/solver.h"
#include "qp/qp_solver.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace nlp {
const char* toString(Status s) noexcept {
    switch(s) {
#define NLP_STATUS(s) case Status::s: return #s
    NLP_STATUS(FirstOrderStationary); NLP_STATUS(IterationLimit); NLP_STATUS(TimeLimit);
    NLP_STATUS(UserStopped); NLP_STATUS(InvalidProblem); NLP_STATUS(EvaluationFailure);
    NLP_STATUS(SubproblemFailure); NLP_STATUS(NoProgress); NLP_STATUS(NumericalFailure);
#undef NLP_STATUS
    }
    return "Unknown";
}
namespace {
using Vec=std::vector<double>;
bool finite(const Vec& v) { for(double a:v) if(!std::isfinite(a)) return false; return true; }
double norm(const Vec& v) { double r=0; for(double a:v) { if(!std::isfinite(a)) return infinity; r=std::max(r,std::abs(a)); } return r; }
double dot(const Vec& a,const Vec& b) { return std::inner_product(a.begin(),a.end(),b.begin(),0.0); }
double violation(double v,Bounds b) { return std::max({0.0,b.lower-v,v-b.upper}); }
void validateBounds(const std::vector<Bounds>& bs) {
    for(auto b:bs) if(std::isnan(b.lower)||std::isnan(b.upper)||b.lower>b.upper||b.lower==infinity||b.upper==-infinity)
        throw std::invalid_argument("invalid bounds");
}
void validateEvaluation(const Evaluation& e,int n,int m) {
    if(e.gradient.size()!=static_cast<size_t>(n)||e.constraints.size()!=static_cast<size_t>(m)||
       e.jacobian.rows()!=m||e.jacobian.columns()!=n||!e.jacobian.validate())
        throw std::invalid_argument("callback derivative dimensions or sparse structure invalid");
    if(!std::isfinite(e.objective)||!finite(e.gradient)||!finite(e.constraints)||!finite(e.jacobian.csrValues()))
        throw std::domain_error("non-finite callback evaluation");
}
struct Triplets {
    Vec r,c,v;
    void add(int row,int col,double value) { if(value!=0) {r.push_back(row); c.push_back(col); v.push_back(value);} }
    qp::SparseMatrix matrix(int rows,int cols) const {return qp::SparseMatrix::fromTriplets(rows,cols,r,c,v);}
};
// Curvature in scaled variable coordinates. Powell damping retains positive
// curvature even for nonconvex objectives and rank-deficient constraints.
class Curvature {
    int n_; bool dense_; Vec b_;
public:
    Curvature(int n,int limit):n_(n),dense_(n<=limit),b_(dense_?static_cast<size_t>(n)*n:n,0) { reset(); }
    void reset() { std::fill(b_.begin(),b_.end(),0); for(int i=0;i<n_;++i) b_[dense_?static_cast<size_t>(i)*n_+i:i]=1; }
    qp::SparseMatrix matrix(int total,double reg) const {
        Triplets t;
        for(int i=0;i<n_;++i) {
            if(dense_) for(int j=0;j<n_;++j) t.add(i,j,b_[static_cast<size_t>(i)*n_+j]+(i==j?reg:0));
            else t.add(i,i,b_[i]+reg);
        }
        for(int i=n_;i<total;++i) t.add(i,i,reg);
        return t.matrix(total,total);
    }
    void update(const Vec& s,Vec y) {
        if(norm(s)<1e-12) return;
        Vec bs(n_,0);
        for(int i=0;i<n_;++i) {
            if(dense_) for(int j=0;j<n_;++j) bs[i]+=b_[static_cast<size_t>(i)*n_+j]*s[j];
            else bs[i]=b_[i]*s[i];
        }
        double sb=dot(s,bs), sy=dot(s,y);
        if(!std::isfinite(sb)||!std::isfinite(sy)||sb<=1e-24) { reset(); return; }
        if(sy<0.2*sb) {
            double theta=0.8*sb/(sb-sy);
            for(int i=0;i<n_;++i) y[i]=theta*y[i]+(1-theta)*bs[i];
            sy=dot(s,y);
        }
        if(!(sy>0)||!std::isfinite(sy)) {reset();return;}
        if(dense_) for(int i=0;i<n_;++i) for(int j=0;j<n_;++j)
            b_[static_cast<size_t>(i)*n_+j]+=y[i]*y[j]/sy-bs[i]*bs[j]/sb;
        else std::fill(b_.begin(),b_.end(),std::clamp(dot(y,y)/sy,1e-6,1e6));
        if(!finite(b_)||norm(b_)>1e12) reset();
    }
};
Vec lagrangian(const Evaluation& e,const Vec& lambda,const Vec& z) {
    Vec g; e.jacobian.transposeMultiply(lambda,g);
    for(size_t j=0;j<g.size();++j) g[j]+=e.gradient[j]+z[j];
    return g;
}
void residuals(Result& r,const Evaluation& e,const std::vector<Bounds>& vb,const std::vector<Bounds>& cb) {
    r.objective=e.objective; r.primalResidual=0; r.complementarity=0;
    auto row=[&](double x,Bounds b,double y) {
        r.primalResidual=std::max(r.primalResidual,violation(x,b));
        double residual=0;
        if(y>0) residual=std::isfinite(b.upper)?std::abs(y*(x-b.upper)):y;
        if(y<0) residual=std::isfinite(b.lower)?std::abs(y*(x-b.lower)):-y;
        if(!std::isfinite(residual)) residual=infinity;
        r.complementarity=std::max(r.complementarity,residual);
    };
    for(size_t j=0;j<vb.size();++j) row(r.primal[j],vb[j],r.boundMultipliers[j]);
    for(size_t i=0;i<cb.size();++i) row(e.constraints[i],cb[i],r.constraintMultipliers[i]);
    r.dualResidual=norm(lagrangian(e,r.constraintMultipliers,r.boundMultipliers));
}
}
Result Solver::solve(const Problem& problem,const Vec& initial,const Options& o) const {
    Result r;
    auto start=std::chrono::steady_clock::now();
    auto elapsed=[&] {return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();};
    auto finish=[&](Status s,const std::string& msg) {r.status=s;r.message=msg;r.solveSeconds=elapsed();return r;};
    try {
        const auto& vb=problem.variableBounds(); const auto& cb=problem.constraintBounds();
        if(vb.size()>1000000||cb.size()>1000000) throw std::invalid_argument("problem exceeds supported index/resource limits");
        int n=static_cast<int>(vb.size()),m=static_cast<int>(cb.size());
        validateBounds(vb);validateBounds(cb);
        if(initial.size()!=vb.size()||!finite(initial)) throw std::invalid_argument("initial point must have n finite entries");
        if(o.iterationLimit<0||o.qpIterationLimit<=0||o.lineSearchLimit<=0||o.denseBfgsLimit<0||o.denseBfgsLimit>2048||
           !std::isfinite(o.tolerance)||o.tolerance<=0||!std::isfinite(o.timeLimitSeconds)||o.timeLimitSeconds<0||
           !std::isfinite(o.initialPenalty)||o.initialPenalty<=0||!std::isfinite(o.maximumPenalty)||o.maximumPenalty<o.initialPenalty||
           !std::isfinite(o.regularization)||o.regularization<=0)
            throw std::invalid_argument("invalid NLP options");
        Vec scale=o.variableScale.empty()?Vec(n,1):o.variableScale;
        if(scale.size()!=vb.size()||!finite(scale)) throw std::invalid_argument("invalid variable scaling");
        for(double v:scale) if(v<=0) throw std::invalid_argument("variable scales must be positive");
        r.primal=initial; r.constraintMultipliers.assign(m,0);r.boundMultipliers.assign(n,0);
        for(int j=0;j<n;++j) r.primal[j]=std::clamp(r.primal[j],vb[j].lower,vb[j].upper);
        auto evaluate=[&](const Vec& x) {++r.evaluations;auto e=problem.evaluate(x);validateEvaluation(e,n,m);return e;};
        Evaluation e;
        try {e=evaluate(r.primal);} catch(const std::domain_error& ex) {return finish(Status::EvaluationFailure,ex.what());}
        r.hasPrimal=true;
        Vec rowScale(m,1);
        if(o.scaleConstraints) {
            const auto& ptr=e.jacobian.csrRowStart();const auto& col=e.jacobian.csrColumnIndex();const auto& val=e.jacobian.csrValues();
            for(int i=0;i<m;++i) {
                double a=1;
                for(auto k=ptr[i];k<ptr[i+1];++k) a=std::max(a,std::abs(val[k]*scale[col[k]]));
                rowScale[i]=std::max(1e-8,1/a);
            }
        }
        auto infeasibility=[&](const Vec& c) {double v=0;for(int i=0;i<m;++i) v+=rowScale[i]*violation(c[i],cb[i]);return v;};
        Curvature b(n,o.denseBfgsLimit); double penalty=o.initialPenalty;
        for(int it=0;;++it) {
            r.iteration=it;r.penalty=penalty;
            residuals(r,e,vb,cb);r.feasible=r.primalResidual<=o.tolerance;
            if(o.callback&&!o.callback(r)) return finish(Status::UserStopped,"stopped by callback");
            if(r.feasible&&r.dualResidual<=o.tolerance&&r.complementarity<=o.tolerance)
                return finish(Status::FirstOrderStationary,"original-unit first-order KKT conditions satisfied; no global or second-order certificate");
            if(o.timeLimitSeconds>0&&elapsed()>=o.timeLimitSeconds) return finish(Status::TimeLimit,"NLP time limit");
            if(it>=o.iterationLimit) return finish(Status::IterationLimit,"NLP iteration limit");
            if(n==0) return finish(Status::NoProgress,"constant constraints violated");
            qp::AdmmResult q; bool elastic=false;
            Vec direction,lambda,z;
            double lastPr=infinity,lastDu=infinity;
            for(int attempt=0;attempt<2;++attempt) {
                elastic=attempt!=0; int nv=n+(elastic?2*m:0);
                qp::QpModel sub;sub.P=b.matrix(nv,o.regularization);sub.q.assign(nv,penalty);
                for(int j=0;j<n;++j) sub.q[j]=e.gradient[j]*scale[j];
                Triplets a;
                const auto& ptr=e.jacobian.csrRowStart();const auto& col=e.jacobian.csrColumnIndex();const auto& val=e.jacobian.csrValues();
                for(int i=0;i<m;++i) {
                    for(auto k=ptr[i];k<ptr[i+1];++k) a.add(i,col[k],val[k]*rowScale[i]*scale[col[k]]);
                    if(elastic) {a.add(i,n+2*i,1);a.add(i,n+2*i+1,-1);}
                    sub.l.push_back((cb[i].lower-e.constraints[i])*rowScale[i]);
                    sub.u.push_back((cb[i].upper-e.constraints[i])*rowScale[i]);
                }
                for(int j=0;j<nv;++j) {
                    a.add(m+j,j,1);
                    sub.l.push_back(j<n?(vb[j].lower-r.primal[j])/scale[j]:0);
                    sub.u.push_back(j<n?(vb[j].upper-r.primal[j])/scale[j]:infinity);
                }
                sub.A=a.matrix(m+nv,nv);
                qp::AdmmOptions qo;qo.iterationLimit=o.qpIterationLimit;qo.threadCount=1;
                qo.primalTolerance=qo.dualTolerance=std::max(1e-14,std::min(1e-10,o.tolerance*0.001)/(1+norm(sub.P.csrValues())));
                qo.terminationCheckFrequency=10;qo.usePolishing=false;
                if(o.timeLimitSeconds>0) {
                    qo.timeLimitSeconds=o.timeLimitSeconds-elapsed();
                    if(qo.timeLimitSeconds<=0) return finish(Status::TimeLimit,"NLP time limit before QP");
                }
                if(elastic) ++r.elasticSubproblems;
                q=qp::QpSolver().solve(sub,qo);r.qpIterations+=q.iterations;
                if(q.status==qp::QpStatus::TimeLimit) return finish(Status::TimeLimit,"time limit in QP");
                if(q.primal.size()!=static_cast<size_t>(nv)||q.constraintDual.size()!=static_cast<size_t>(m+nv)||!finite(q.primal)||!finite(q.constraintDual)) continue;
                // Independently gate the inner solve in its original (SQP) units.
                Vec ax,px,aty;sub.A.multiply(q.primal,ax);sub.P.multiply(q.primal,px);sub.A.transposeMultiply(q.constraintDual,aty);
                double pr=0;for(int i=0;i<m+nv;++i) pr=std::max(pr,violation(ax[i],{sub.l[i],sub.u[i]}));
                for(int j=0;j<nv;++j) px[j]+=sub.q[j]+aty[j];
                lastPr=pr;lastDu=norm(px);
                if(q.status!=qp::QpStatus::Optimal||pr>std::max(1e-7,o.tolerance*0.1)||norm(px)>std::max(1e-7,o.tolerance*0.1)) continue;
                direction.assign(q.primal.begin(),q.primal.begin()+n);lambda.resize(m);z.resize(n);
                for(int i=0;i<m;++i) lambda[i]=q.constraintDual[i]*rowScale[i];
                for(int j=0;j<n;++j) z[j]=q.constraintDual[m+j]/scale[j];
                break;
            }
            if(direction.empty()) return finish(Status::SubproblemFailure,std::string("QP failed independent residual validation: ")+qp::toString(q.status)+", primal="+std::to_string(lastPr)+", dual="+std::to_string(lastDu)+" (not an NLP infeasibility certificate)");
            r.constraintMultipliers=lambda;r.boundMultipliers=z;
            residuals(r,e,vb,cb);r.feasible=r.primalResidual<=o.tolerance;
            if(r.feasible&&r.dualResidual<=o.tolerance&&r.complementarity<=o.tolerance)
                return finish(Status::FirstOrderStationary,"original-unit first-order KKT conditions satisfied; no global or second-order certificate");
            double dualNorm=0;for(int i=0;i<m;++i) dualNorm=std::max(dualNorm,std::abs(lambda[i]/rowScale[i]));
            penalty=std::min(o.maximumPenalty,std::max(penalty,1.5*dualNorm+1));
            Vec step(n);for(int j=0;j<n;++j) step[j]=direction[j]*scale[j];
            Vec jd;e.jacobian.multiply(step,jd);for(int i=0;i<m;++i) jd[i]+=e.constraints[i];
            double v=infeasibility(e.constraints), linearV=infeasibility(jd);
            double predicted=dot(e.gradient,step)+penalty*(linearV-v);
            if(!std::isfinite(predicted)||!finite(step)) return finish(Status::NumericalFailure,"non-finite merit model");
            if(predicted>=-1e-16) {
                if(penalty<o.maximumPenalty&&v>o.tolerance) {penalty=std::min(o.maximumPenalty,penalty*10);b.reset();continue;}
                return finish(Status::NoProgress,"no merit descent; possible degeneracy or local infeasibility, not a proof");
            }
            Vec trial(n);Evaluation next;double alpha=1;bool accepted=false;
            for(int ls=0;ls<o.lineSearchLimit;++ls,alpha*=0.5) {
                if(o.timeLimitSeconds>0&&elapsed()>=o.timeLimitSeconds) return finish(Status::TimeLimit,"time limit in line search");
                for(int j=0;j<n;++j) trial[j]=std::clamp(r.primal[j]+alpha*step[j],vb[j].lower,vb[j].upper);
                try {
                    next=evaluate(trial);
                    double change=(next.objective-e.objective)+penalty*(infeasibility(next.constraints)-v);
                    if(std::isfinite(change)&&change<=1e-4*alpha*predicted) {accepted=true;break;}
                } catch(const std::domain_error&) { /* Backtrack into the evaluation domain. */ }
                ++r.rejectedTrials;
            }
            if(!accepted) return finish(Status::NoProgress,"merit line search exhausted; last valid iterate retained");
            Vec oldG=lagrangian(e,lambda,z),newG=lagrangian(next,lambda,z),s(n),y(n);
            for(int j=0;j<n;++j) {s[j]=(trial[j]-r.primal[j])/scale[j];y[j]=(newG[j]-oldG[j])*scale[j];}
            b.update(s,y);r.primal=std::move(trial);e=std::move(next);r.stepLength=alpha;
        }
    } catch(const std::invalid_argument& ex) {return finish(Status::InvalidProblem,ex.what());}
      catch(const std::exception& ex) {return finish(Status::NumericalFailure,ex.what());}
}
} // namespace nlp
