#pragma once

#include <cstddef>
#include <cmath>
#include <limits>
#include <vector>

#include "sqlite_storage.hpp"

/*
┌───────────────────────────────────────────────────┐
│   Study - context matrix & windowing              │
└───────────────────────────────────────────────────┘
*/

inline double SafeLog(double v) {
    return (v > 0.0 && std::isfinite(v)) ? std::log(v) : std::numeric_limits<double>::quiet_NaN();
}

std::vector<std::vector<double>> BuildMatrix(const SQLite::PrepData& p);


void ReportNaN(const std::vector<std::vector<double>>& m);

std::vector<std::vector<double>> Window(const std::vector<std::vector<double>>& m, size_t t, size_t len);
