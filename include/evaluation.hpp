#pragma once

#include <cstddef>
#include <cmath>
#include <limits>
#include <vector>

// check thoroughly 

inline double QLike(double logActual, double logPred) {
    double r = std::exp(logActual - logPred);
    return r - (logActual - logPred) - 1.0;
}

/* ---- HAR-RV baseline -------------------------------------------------- */

// backward-looking mean of x over [t-len+1, t]. NaN if any element is NaN.
inline double TrailMean(const std::vector<double>& x, size_t t, size_t len) {
    if (t + 1 < len) return std::numeric_limits<double>::quiet_NaN();
    double s = 0.0;
    for (size_t k = t + 1 - len; k <= t; ++k) {
        if (!std::isfinite(x[k])) return std::numeric_limits<double>::quiet_NaN();
        s += x[k];
    }
    return s / (double)len;
}

// HAR design row at origin t: [1, daily, weekly, monthly] off the log-RV row.
// all three look backward only, so using them at t leaks nothing.
inline bool HarRow(const std::vector<double>& logrv, size_t t, double out[4]) {
    out[0] = 1.0;
    out[1] = logrv[t];
    out[2] = TrailMean(logrv, t, 5);
    out[3] = TrailMean(logrv, t, 22);
    for (int j = 0; j < 4; ++j) if (!std::isfinite(out[j])) return false;
    return true;
}

// OLS by normal equations over origins [begin, end).  X'X b = X'y, 4x4.
bool HarFit(const std::vector<double>& logrv, const std::vector<double>& y,
            size_t begin, size_t end, double beta[4]);

inline double HarPredict(const double beta[4], const double x[4]) {
    return beta[0]*x[0] + beta[1]*x[1] + beta[2]*x[2] + beta[3]*x[3];
}

/* ---- Diebold-Mariano --------------------------------------------------- */
// Bartlett/Newey-West long-run variance + Harvey-Leybourne-Newbold correction,
// p from t(n-1).  stat < 0 means lossA is the LOWER loss, i.e. A wins.

struct DMResult { double stat, pvalue; size_t n; };

DMResult DieboldMariano(const std::vector<double>& lossA, const std::vector<double>& lossB, size_t h);