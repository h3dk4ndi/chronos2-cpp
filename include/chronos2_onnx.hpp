#pragma once

#include <onnxruntime/onnxruntime_cxx_api.h>

#include <cstdint>
#include <string>
#include <vector>

/*
┌───────────────────────────────────────────────────┐
│   Chronos-2                                       │
└───────────────────────────────────────────────────┘
*/

// check thoroughly !!!

class Chronos2ONNX {
private:
    Ort::Env            env;        // declaration order == construction order:
    Ort::SessionOptions options;    // env and options must come before session
    Ort::Session        session;
    Ort::MemoryInfo     mem;

    std::vector<float>  buf;                    // flat context; must outlive Run()
    int64_t nq = 0, H = 0, medianIdx = 0;       // read off the graph, not hardcoded

    // SetIntraOpNumThreads has to happen BEFORE the session exists, and an
    // init-list gives you no body to call it from -> build the options here.
    static Ort::SessionOptions makeOptions() {
        Ort::SessionOptions o;
        o.SetIntraOpNumThreads(1);              // 1 -> bitwise reproducible
        o.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        return o;
    }

#ifdef _WIN32
    static std::wstring toOrt(const std::string& s) { return std::wstring(s.begin(), s.end()); }
#else
    static std::string  toOrt(const std::string& s) { return s; }
#endif

public:
    struct Forecast { std::vector<double> median, lo, hi; };   // length H each

    explicit Chronos2ONNX(const std::string& modelPath);

    int64_t horizon()   const { return H;  }
    int64_t quantiles() const { return nq; }

    // rows[0] = target (raw log-RV), rows[1..] = covariates, same order every call
    Forecast predict(const std::vector<std::vector<double>>& rows);

    // the study's point forecast: mean of the median path
    double pointForecast(const std::vector<std::vector<double>>& rows);
};