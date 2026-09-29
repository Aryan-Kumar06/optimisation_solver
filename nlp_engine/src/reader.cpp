#include "nlp/model.h"
#include <istream>
#include <stdexcept>
#include <unordered_map>

namespace nlp {
Input read(std::istream& in) {
    auto token=[&]() {std::string s;if(!(in>>s)) throw std::invalid_argument("unexpected end of NLP input");return s;};
    auto expect=[&](const char* s) {if(token()!=s) throw std::invalid_argument(std::string("expected ")+s);};
    auto integer=[&](int maximum) {auto s=token();size_t used=0;long v=std::stol(s,&used);if(used!=s.size()||v<0||v>maximum) throw std::invalid_argument("NLP integer out of range");return static_cast<int>(v);};
    auto number=[&]() {auto s=token();size_t used=0;double v=std::stod(s,&used);if(used!=s.size()||std::isnan(v)) throw std::invalid_argument("invalid NLP number");return v;};
    expect("nlp");expect("1");expect("variables");int n=integer(1000000);
    std::vector<Bounds> vb(n);std::vector<double> initial(n);
    for(int j=0;j<n;++j) {vb[j]={number(),number()};initial[j]=number();}
    expect("nodes");int count=integer(1000000);std::vector<Expression> nodes;nodes.reserve(count);
    auto ref=[&]() {int i=integer(count);if(i>=static_cast<int>(nodes.size())) throw std::invalid_argument("expression reference must name an earlier node");return nodes[i];};
    for(int k=0;k<count;++k) {
        std::string op=token();
        if(op=="const") nodes.emplace_back(number());
        else if(op=="var") {int j=integer(n);if(j>=n) throw std::invalid_argument("variable index out of range");nodes.push_back(variable(j));}
        else if(op=="add"||op=="sub"||op=="mul"||op=="div") {
            auto a=ref(),b=ref();
            nodes.push_back(op=="add"?a+b:op=="sub"?a-b:op=="mul"?a*b:a/b);
        } else {
            if(op!="neg"&&op!="exp"&&op!="log"&&op!="sin"&&op!="cos"&&op!="sqrt"&&op!="square")
                throw std::invalid_argument("unknown NLP operation: "+op);
            auto a=ref();
            if(op=="neg") nodes.push_back(-a);
            if(op=="exp") nodes.push_back(exp(a));
            if(op=="log") nodes.push_back(log(a));
            if(op=="sin") nodes.push_back(sin(a));
            if(op=="cos") nodes.push_back(cos(a));
            if(op=="sqrt") nodes.push_back(sqrt(a));
            if(op=="square") nodes.push_back(square(a));
        }
    }
    expect("objective");auto objective=ref();expect("constraints");int m=integer(1000000);
    std::vector<Expression> constraints;std::vector<Bounds> cb;
    for(int i=0;i<m;++i) {double l=number(),u=number();constraints.push_back(ref());cb.push_back({l,u});}
    std::string trailing;if(in>>trailing) throw std::invalid_argument("unexpected trailing NLP input");
    return {Model(std::move(vb),objective,std::move(constraints),std::move(cb)),std::move(initial)};
}
} // namespace nlp
