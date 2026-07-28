# chronos2-cpp

End-to-end C++ pipeline for realised-volatility forecasting: Bloomberg BLPAPI
ingest → SQLite → feature engineering → Chronos-2 inference via ONNX Runtime,
benchmarked against HAR-RV and persistence with Diebold-Mariano tests.

## Layout

| Path | Contents |
|---|---|
| `include/config.hpp` | `ROLL_W`, `CONTEXT`, `TEST_FRAC` |
| `include/types.hpp` | `InstrumentMeta` |
| `include/sqlite_storage.hpp` | `SQLite` — `blp_data`, `instrument_meta`, `prep_data` |
| `include/bloomberg_client.hpp` | `Bloomberg` — historical + reference data requests |
| `include/stationarity.hpp` | `Adfuller` (Eigen), `FracDiff` (de Prado) |
| `include/split.hpp` | purged chronological train/test split |
| `include/preprocessing.hpp` | rolling primitives + 5 RV estimators + label |
| `include/chronos2_onnx.hpp` | `Chronos2ONNX` — ORT session wrapper |
| `include/study.hpp` | context matrix assembly and windowing |
| `include/evaluation.hpp` | QLIKE, HAR-RV, Diebold-Mariano |

## Requirements

- C++17
- Eigen 3
- SQLite 3
- ONNX Runtime 1.26+
- Bloomberg BLPAPI (optional — only for `-DUSE_BLPAPI=ON`)

## Model weights

`models/chronos2.onnx` ships in the repo. The external weights file
(~107 MB) exceeds GitHub's file limit and is **not** committed — download it
separately and place it beside the graph as `models/chronos2.onnx.data`.
The filename is recorded inside the graph; the ORT session constructor throws
if it does not match exactly.

## Build

    cmake -B build -DONNXRUNTIME_ROOT=/path/to/onnxruntime
    cmake --build build

Add `-DUSE_BLPAPI=ON -DBLPAPI_ROOT=/path/to/blpapi` to compile the ingest path.
Without it the binary runs from an existing `blp.db` and needs no Terminal.

## Results

XAU Curncy, 2,321 origins, `ROLL_W = 21`, `CONTEXT = 512`:

| Model | QLIKE |
|---|---|
| Chronos-2 (last of path) | 0.2961 |
| Chronos-2 (mean of path) | 0.3068 |
| HAR-RV | 0.2760 |
| Persistence | 0.3308 |

Chronos-2 vs persistence: DM = −3.75, p = 1.8e-4. Chronos-2 vs HAR-RV is not
statistically distinguishable at these settings.
