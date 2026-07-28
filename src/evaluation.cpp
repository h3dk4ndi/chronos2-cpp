#include "evaluation.hpp"

#include <eigen3/Eigen/Dense>

namespace {

double StudentTwoSidedP(double t, double v) {
    double x = v / (v + t*t), a = v/2.0, b = 0.5;
    auto betacf = [](double a, double b, double x) {
        double qab=a+b, qap=a+1.0, qam=a-1.0, c=1.0, d=1.0-qab*x/qap;
        if (std::fabs(d) < 1e-300) d = 1e-300;
        d = 1.0/d; double h = d;
        for (int m = 1; m <= 300; ++m) {
            int m2 = 2*m;
            double aa = m*(b-m)*x/((qam+m2)*(a+m2));
            d = 1.0+aa*d; c = 1.0+aa/c;
            if (std::fabs(d)<1e-300) d=1e-300;
            if (std::fabs(c)<1e-300) c=1e-300;
            d = 1.0/d; h *= d*c;
            aa = -(a+m)*(qab+m)*x/((a+m2)*(qap+m2));
            d = 1.0+aa*d; c = 1.0+aa/c;
            if (std::fabs(d)<1e-300) d=1e-300;
            if (std::fabs(c)<1e-300) c=1e-300;
            d = 1.0/d; double del = d*c; h *= del;
            if (std::fabs(del-1.0) < 3e-14) break;
        }
        return h;
    };
    double lbeta = std::lgamma(a+b) - std::lgamma(a) - std::lgamma(b);
    if (x < (a+1.0)/(a+b+2.0))
        return std::exp(lbeta + a*std::log(x) + b*std::log(1.0-x)) * betacf(a,b,x) / a;
    return 1.0 - std::exp(lbeta + b*std::log(1.0-x) + a*std::log(x)) * betacf(b,a,1.0-x) / b;
}

}   // namespace

// OLS by normal equations over origins [begin, end).  X'X b = X'y, 4x4.
bool HarFit(const std::vector<double>& logrv, const std::vector<double>& y,
            size_t begin, size_t end, double beta[4]) {
    Eigen::Matrix4d XtX = Eigen::Matrix4d::Zero();
    Eigen::Vector4d Xty = Eigen::Vector4d::Zero();
    double x[4];
    size_t used = 0;
    for (size_t t = begin; t < end; ++t) {
        if (!std::isfinite(y[t]) || !HarRow(logrv, t, x)) continue;
        Eigen::Map<Eigen::Vector4d> v(x);
        XtX += v * v.transpose();
        Xty += v * y[t];
        ++used;
    }
    if (used < 100) return false;
    Eigen::Vector4d b = XtX.ldlt().solve(Xty);
    if (!b.allFinite()) return false;
    for (int j = 0; j < 4; ++j) beta[j] = b[j];
    return true;
}

DMResult DieboldMariano(const std::vector<double>& lossA, const std::vector<double>& lossB, size_t h) {
    const size_t n = lossA.size();
    std::vector<double> d(n);
    double dbar = 0.0;
    for (size_t i = 0; i < n; ++i) { d[i] = lossA[i] - lossB[i]; dbar += d[i]; }
    dbar /= (double)n;

    const size_t L = (h > 1) ? h - 1 : 0;         // overlapping forecasts -> MA(h-1)
    double var = 0.0;
    for (size_t j = 0; j <= L; ++j) {
        double g = 0.0;
        for (size_t i = j; i < n; ++i) g += (d[i]-dbar)*(d[i-j]-dbar);
        g /= (double)(n - j);
        var += (j == 0) ? g : 2.0*(1.0 - (double)j/(double)(L+1))*g;
    }

    double stat = dbar / std::sqrt(var / (double)n);
    const double nn = (double)n, hh = (double)h;
    stat *= std::sqrt((nn + 1.0 - 2.0*hh + hh*(hh-1.0)/nn) / nn);   // HLN

    DMResult r; r.n = n; r.stat = stat;
    r.pvalue = StudentTwoSidedP(stat, nn - 1.0);
    return r;
}