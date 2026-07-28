#include "chronos2_onnx.hpp"

#include <cstdio>
#include <stdexcept>

Chronos2ONNX::Chronos2ONNX(const std::string& modelPath)
    : env(ORT_LOGGING_LEVEL_WARNING, "chronos2"),
      options(makeOptions()),
      session(env, toOrt(modelPath).c_str(), options),     // throws if the
      mem(Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU))
{                                                          // .data file is missing
    std::vector<int64_t> shp = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    if (shp.size() != 3) throw std::runtime_error("unexpected output rank");
    nq        = shp[1];        // 13
    H         = shp[2];        // 21
    medianIdx = nq / 2;        // 6
    printf("[chronos2] quantiles=%lld horizon=%lld median_idx=%lld\n", (long long)nq, (long long)H, (long long)medianIdx);
}

Chronos2ONNX::Forecast Chronos2ONNX::predict(const std::vector<std::vector<double>>& rows) {
    if (rows.empty()) throw std::runtime_error("no rows");

    const int64_t V = (int64_t)rows.size();
    const int64_t L = (int64_t)rows[0].size();
    if (L < 64) throw std::runtime_error("context shorter than 64");
    for (size_t v = 0; v < rows.size(); ++v)
        if ((int64_t)rows[v].size() != L) throw std::runtime_error("ragged context");

    buf.resize((size_t)(V * L));                    // double -> float, row-major
    for (int64_t v = 0; v < V; ++v)
        for (int64_t l = 0; l < L; ++l)
            buf[(size_t)(v * L + l)] = (float)rows[v][l];

    int64_t shape[2] = { V, L };
    Ort::Value in = Ort::Value::CreateTensor<float>(mem, buf.data(), buf.size(), shape, 2);
    // ^ this WRAPS buf, it does not copy it. that is why buf is a member.

    const char* inNames[]  = { "context"   };
    const char* outNames[] = { "quantiles" };
    std::vector<Ort::Value> out =
        session.Run(Ort::RunOptions{nullptr}, inNames, &in, 1, outNames, 1);

    const float* q = out[0].GetTensorData<float>();   // flat [V, nq, H]

    Forecast f;
    f.median.resize(H); f.lo.resize(H); f.hi.resize(H);
    for (int64_t h = 0; h < H; ++h) {
        f.median[h] = (double)q[(0 * nq + medianIdx) * H + h];   // row 0, q0.5
        f.lo[h]     = (double)q[(0 * nq + 2        ) * H + h];   // row 0, q0.1
        f.hi[h]     = (double)q[(0 * nq + 10       ) * H + h];   // row 0, q0.9
    }
    return f;
}

// the study's point forecast: mean of the median path
double Chronos2ONNX::pointForecast(const std::vector<std::vector<double>>& rows) {
    Forecast f = predict(rows);
    double s = 0.0;
    for (size_t i = 0; i < f.median.size(); ++i) s += f.median[i];
    return s / (double)f.median.size();
}