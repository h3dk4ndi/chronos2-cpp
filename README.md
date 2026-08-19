# chronos2-cpp

<p align="center">
  <img src="images/ceo_mascot.png" alt="Little CEO mascot" width="250">
</p>

End-to-end C++ pipeline for realised-volatility forecasting: Bloomberg BLPAPI
ingest → SQLite → feature engineering → Chronos-2 inference via ONNX Runtime,
benchmarked against a from-scratch TCN, HAR-RV and persistence with
Diebold-Mariano tests.

## Contributors

Special thanks to [@ByteJoseph](https://github.com/ByteJoseph) for their contributions, feedback, and support during the development of this project.

Additional contributors and their specific contributions will be acknowledged here as the project evolves.

## Layout

| Path | Contents |
|---|---|
| `include/config.hpp` | `ROLL_W`, `CONTEXT`, `TEST_FRAC` |
| `include/types.hpp` | `InstrumentMeta` |
| `include/sqlite_storage.hpp` | `SQLite` — `blp_data`, `instrument_meta`, `prep_data` |
| `include/bloomberg_client.hpp` | `Bloomberg` — historical + reference data requests |
| `include/stationarity.hpp` | `Adfuller` (Eigen), `FracDiff` (de Prado) |
| `include/split.hpp` | purged chronological train/val/test split |
| `include/preprocessing.hpp` | rolling primitives + RV estimators, semivariance, jumps, leverage |
| `include/chronos2_onnx.hpp` | `Chronos2ONNX` — ORT session wrapper |
| `include/study.hpp` | context matrix assembly and windowing |
| `include/evaluation.hpp` | QLIKE, HAR-RV, Diebold-Mariano |
| `tcnvol/` | NumPy TCN — layers, weight-norm dilated causal convs, AdamW, trainer |
| `run_tcn.py` | TCN entry point; writes per-origin QLIKE to `tcn_loss.csv` |

## Features

`BuildMatrix()` assembles a 14-row context matrix. Row 0 is the target series;
rows 1-13 are covariates supplied to Chronos-2 through group attention and to
the TCN as input channels.

| Rows | Contents | Transform |
|---|---|---|
| 0 | close-to-close RV (target) | log |
| 1-4 | Parkinson, Garman-Klass, Rogers-Satchell, Yang-Zhang | log |
| 5 | returns | raw |
| 6-7 | negative / positive realised semivariance | log |
| 8 | bipower variation | log |
| 9-13 | signed jump, leverage, 5d leverage mean, jump component, relative jump | raw |

Rows 9-13 can be zero or negative, so they are passed unlogged.

## Requirements

- C++17
- Eigen 3
- SQLite 3
- ONNX Runtime 1.26+
- Bloomberg BLPAPI 
- Python 3 + NumPy (TCN only)

## Model weights

`models/chronos2.onnx` ships in the repo. The external weights file
(~478 MB for the 120M base) exceeds GitHub's file limit and is **not**
committed — download it separately and place it beside the graph as
`models/chronos2.onnx.data`. The filename is recorded inside the graph;
the ORT session constructor throws if it does not match exactly.

`models/onnx_chronos_v2.ipynb` regenerates the graph. Point `MODEL` at a local
folder to export fine-tuned weights instead of the base checkpoint; nothing
else in the notebook changes.

## Run

    ./chronos2.exe        # Chronos-2, HAR-RV, persistence walk-forward
    python run_tcn.py     # TCN — trains, evaluates, writes tcn_loss.csv

Both read `prep_data` from `blp.db` and score the same 2,321 origins.

## Results

XAU Curncy, 2,321 origins, `ROLL_W = 21`, `CONTEXT = 512`, 14 covariate rows.
Split: 4,326 train / 523 validation / 2,321 test windows, purged by 21 days at
each boundary. Chronos-2 is the 120M `amazon/chronos-2` checkpoint.

| Model | QLIKE | Skill vs HAR |
|---|---|---|
| **TCN (from scratch)** | **0.2441** | **+11.5%** |
| HAR-RV | 0.2760 | — |
| Chronos-2 zero-shot (last of path) | 0.2787 | −1.0% |
| Chronos-2 fine-tuned (mean of path) | 0.2814 | −2.0% |
| Chronos-2 zero-shot (mean of path) | 0.2912 | −5.5% |
| Persistence | 0.3308 | −19.9% |
| Chronos-2 fine-tuned (last of path) | 0.3370 | −22.1% |

Diebold-Mariano, Newey-West with the Harvey-Leybourne-Newbold small-sample
correction. Positive statistic favours the second model.

| Comparison | DM | p |
|---|---|---|
| Chronos-2 last vs HAR-RV | +0.13 | 0.90 |
| Chronos-2 mean vs HAR-RV | +0.60 | 0.55 |
| Chronos-2 last vs mean | −1.18 | 0.24 |
| Chronos-2 mean vs persistence | −4.04 | 5.6e-05 |

### Notes

Zero-shot Chronos-2 is statistically indistinguishable from HAR-RV
(DM = 0.13, p = 0.90) despite never having seen gold. It beats persistence
decisively.

Fine-tuning was attempted with LoRA on the attention projections
(2.4M of 122M parameters trainable) over 4,911 purged training windows. The
objective was pinball loss averaged across all 21 quantiles and all 21 horizon
steps — the pretraining objective, not the evaluation target. Path-mean
accuracy improved 3.4%; terminal-step accuracy, which is what QLIKE measures
here, degraded 21%. The training objective has to match the evaluation target.

The TCN is a 7-block weight-normalised dilated causal network, dilations
`[1,2,4,8,16,32,64]` giving a receptive field of 509, roughly 8.6k parameters
against Chronos-2's 121.8M. It trains from scratch on the same 4,326 windows
that were insufficient to adapt the foundation model.

### Caveats

- The TCN result is a single seed. A multi-seed run and a Diebold-Mariano test
  against HAR-RV are pending.
- The covariate ablation (6 / 8 / 14 rows) was run on `chronos-2-small`; only
  the 14-row specification has been evaluated on the 120M checkpoint.
