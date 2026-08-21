#include "study.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

// 18/08/2026 -- BuildMatrix is no longer hradcoded in terms of variable quantity
/*
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
*/

std::vector<std::vector<double>> BuildMatrix(
    SQLite& sql,
    const std::vector<std::string>& securities,
    const std::string& targetSecurity,
    std::size_t trainEnd)
{
    // 1. Load XAU and use its dates as the master calendar.
    SQLite::PrepData target = sql.loadPrep(targetSecurity);
    const auto& masterDates = target.dates;
    const std::size_t N = masterDates.size();

    std::vector<std::vector<double>> M;
    M.reserve(securities.size() * 14);

    // 2. Ensure XAU is the first security.
    std::vector<std::string> ordered{targetSecurity};

    for (const auto& sec : securities) {
        if (sec != targetSecurity)
            ordered.push_back(sec);
    }

    // 3. Load and append 14 aligned features per security.
    for (const auto& sec : ordered) {
        SQLite::PrepData p =
            sec == targetSecurity ? target : sql.loadPrep(sec);

        auto appendAligned =
            [&](const std::vector<double>& source, bool takeLog)
        {
            std::vector<double> row(
                N,
                std::numeric_limits<double>::quiet_NaN()
            );

            std::size_t i = 0; // XAU date
            std::size_t j = 0; // current security date

            while (i < N && j < p.dates.size()) {
                if (masterDates[i] == p.dates[j]) {
                    row[i] = takeLog
                        ? SafeLog(source[j])
                        : source[j];

                    ++i;
                    ++j;
                }
                else if (p.dates[j] < masterDates[i]) {
                    ++j;
                }
                else {
                    // No observation for this security on XAU date i.
                    ++i;
                }
            }

            M.push_back(std::move(row));
        };

        appendAligned(p.close2closeRV,           true);
        appendAligned(p.parkinson,               true);
        appendAligned(p.garmanKlass,             true);
        appendAligned(p.rogersSatchell,          true);
        appendAligned(p.yangZhang,               true);
        appendAligned(p.returns,                 false);
        appendAligned(p.negativeRealisedSemivar, true);
        appendAligned(p.positiveRealisedSemivar, true);
        appendAligned(p.bipowerVariation,        true);
        appendAligned(p.signedJump,              false);
        appendAligned(p.leverage,                false);
        appendAligned(p.leverageMean5,           false);
        appendAligned(p.jumpComponent,           false);
        appendAligned(p.relativeJump,            false);
    }

    if (M.size() != securities.size() * 14)
        throw std::runtime_error("Incorrect matrix width.");

    return M;
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