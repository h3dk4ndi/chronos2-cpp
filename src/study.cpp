#include "study.hpp"

#include <cmath>
#include <cstdio>
#include <limits>

std::vector<std::vector<double>> BuildMatrix(const SQLite::PrepData& p) {
    const size_t N = p.dates.size();
    std::vector<std::vector<double>> m(6, std::vector<double>(N));
    for (size_t i = 0; i < N; ++i) {
        m[0][i] = SafeLog(p.close2closeRV[i]);      // 0  TARGET
        m[1][i] = SafeLog(p.parkinson[i]);
        m[2][i] = SafeLog(p.garmanKlass[i]);
        m[3][i] = SafeLog(p.rogersSatchell[i]);
        m[4][i] = SafeLog(p.yangZhang[i]);
        m[5][i] = p.returns[i];                     // already a rate, no log
    }
    return m;
}

void ReportNaN(const std::vector<std::vector<double>>& m) {
    for (size_t v = 0; v < m.size(); ++v) {
        size_t n = 0;
        for (size_t i = 0; i < m[v].size(); ++i) if (!std::isfinite(m[v][i])) ++n;
        printf("   row %zu : %6zu / %6zu NaN (%.2f%%)\n",
               v, n, m[v].size(), 100.0 * (double)n / (double)m[v].size());
    }
}

std::vector<std::vector<double>> Window(const std::vector<std::vector<double>>& m, size_t t, size_t len) {
    std::vector<std::vector<double>> w(m.size(), std::vector<double>(len));
    for (size_t v = 0; v < m.size(); ++v)
        for (size_t k = 0; k < len; ++k)
            w[v][k] = m[v][t + 1 - len + k];
    return w;
}