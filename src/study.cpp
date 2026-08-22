#include "study.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

// 18/08/2026 -- BuildMatrix is no longer hardcoded in terms of variable quantity
std::vector<std::vector<double>> BuildMatrix(const SQLite::PrepData& p) {
    const size_t N = p.dates.size();
    std::vector<std::vector<double>> m;

    auto addLog = [&](const std::vector<double>& src) {
        std::vector<double> row(N);
        for (size_t i = 0; i < N; ++i) row[i] = SafeLog(src[i]);
        m.push_back(std::move(row));
    };

    addLog(p.close2closeRV);              // 0  TARGET
    addLog(p.parkinson);                  // 1
    addLog(p.garmanKlass);                // 2
    addLog(p.rogersSatchell);             // 3
    addLog(p.yangZhang);                  // 4
    m.push_back(p.returns);               // 5  already a rate, no log
    addLog(p.negativeRealisedSemivar);    // 6
    addLog(p.positiveRealisedSemivar);    // 7
    addLog(p.bipowerVariation);      // 8   variance, > 0
    m.push_back(p.signedJump);       // 9   can be negative
    m.push_back(p.leverage);         // 10  <= 0
    m.push_back(p.leverageMean5);    // 11  <= 0
    m.push_back(p.jumpComponent);    // 12  >= 0, exactly 0 on most days
    m.push_back(p.relativeJump);     // 13  [0,1)

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
