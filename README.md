# chronos2-cpp

<p align="center">
  <img src="images/ceo_mascot.png" alt="Little CEO mascot" width="250">
</p>

End-to-end C++ pipeline for realised-volatility forecasting: Bloomberg BLPAPI
ingest → SQLite → feature engineering → Chronos-2 inference via ONNX Runtime,
benchmarked against a from-scratch TCN, HAR-RV and persistence with
Diebold-Mariano tests.

## Installation

The project is currently configured for **Windows using MSYS2 UCRT64**, GCC/G++, CMake, Ninja, Eigen3, SQLite, ONNX Runtime, and the Bloomberg BLPAPI C++ SDK.

### 1. Clone the repository

```bash
git clone https://github.com/h3dk4ndi/chronos2-cpp.git
cd chronos2-cpp
```

Or manually:
- Open the repository in GitHub.
- Click **Code**.
- Select **Download ZIP**.
- Extract the ZIP archive.
- Open the extracted `chronos2-cpp` folder in VS Code.


### 2. Install MSYS2 UCRT64 dependencies
Open an MSYS2 UCRT64 terminal; update the MSYS2 environment:

```bash
pacman -Syu
```
If MSYS2 asks you to close the terminal while updating core packages, close it, reopen the UCRT64 terminal, and run again:
```bash
pacman -Syu
```
You ideally want something like:
```
:: Synchronizing package databases...
ucrt64 is up to date
mingw64 is up to date
clang64 is up to date
msys is up to date

:: Starting core system upgrade...
 there is nothing to do
:: Starting full system upgrade...
 there is nothing to do
```
 
Install the required packages: 
```bash
pacman -S mingw-w64-ucrt-x86_64-cmake \
          mingw-w64-ucrt-x86_64-ninja \
          mingw-w64-ucrt-x86_64-eigen3 \
          mingw-w64-ucrt-x86_64-sqlite3 \
          mingw-w64-ucrt-x86_64-onnxruntime
```
Verify the development environment:
```bash
gcc --version
g++ --version
cmake --version
ninja --version
```
You should see version information similar to:
```
gcc.exe (Rev8, Built by MSYS2 project) 16.x.x
g++.exe (Rev8, Built by MSYS2 project) 16.x.x
cmake version 4.x.x
1.x.x
```
### 3. Verify ONNX Runtime
The ONNX Runtime C++ headers should be installed under the UCRT64 environment. Run: 
```bash
find /ucrt64 -name "onnxruntime_cxx_api.h"
```
A correct installation should return:
```/ucrt64/include/onnxruntime/onnxruntime_cxx_api.h```
The corresponding runtime DLLs should also exist:
```
/ucrt64/bin/onnxruntime.dll
/ucrt64/bin/onnxruntime_providers_shared.dll
```

### 4. Download the Bloomberg BLPAPI C++ SDK

Download the Bloomberg BLPAPI C++ SDK for Windows:
[Bloomberg SDK C++ for Windows](https://blpapi.bloomberg.com/download/releases/raw/files/blpapi_cpp_3.26.7.1-windows.zip) 

Extract the archive. For version 3.26.7.1, the resulting SDK directory should have approximately the following structure:
```
blpapi_cpp_3.26.7.1/
├── include/
│   ├── blpapi_session.h
│   └── ...
└── lib/
    ├── blpapi3_64.dll
    ├── blpapi3_64.lib
    ├── blpapi3_32.dll
    └── blpapi3_32.lib
```
The Bloomberg SDK does not need to be installed system-wide or copied into the repository. The build system only needs the location of the extracted SDK.

> The BLPAPI SDK provides the API libraries only. Bloomberg market-data retrieval still requires access to Bloomberg Professional / Bloomberg Terminal with an active Bloomberg session.

### 5. Configure local paths in build.bat

Open ```build.bat``` and set the paths to your local UCRT64 and Bloomberg SDK installations.
For example: 
```
set "UCRT64=C:\path\to\msys64\ucrt64"

set "BLPAPI_ROOT=C:\Users\yourName\Downloads\blpapi_cpp_3.26.7.1-windows\blpapi_cpp_3.26.7.1"
```
BLPAPI_ROOT must point to the directory that directly contains:
```
include\
lib\
```

### 6. Configure VS Code IntelliSense
In VS Code press: 
``` Ctrl + Shift + P ```
and select: 
``` C/C++: Edit Configurations (JSON) ```

Configure ```.vscode/c_cpp_properties.json``` using the actual location of your UCRT64 installation:

```bash
{
    "configurations": [
        {
            "name": "UCRT64",
            "compilerPath": "C:/path/to/msys64/ucrt64/bin/g++.exe",
            "includePath": [
                "${workspaceFolder}/**",
                "C:/path/to/msys64/ucrt64/include",
                "C:/path/to/msys64/ucrt64/include/onnxruntime"
            ],
            "cppStandard": "c++17",
            "cStandard": "c17",
            "intelliSenseMode": "windows-gcc-x64"
        }
    ],
    "version": 4
}
```

If VS Code asks which compiler should be used for IntelliSense, select the UCRT64 ```g++.exe``` installation.

If stale include errors remain:
```
Ctrl + Shift + P
→ C/C++: Reset IntelliSense Database
```
and then reload the VS Code window.

Additionally, configure the default C++ compiler used by the VS Code C/C++ extension (VS Code's ```settings.json```, not ```c_cpp_properties.json```). Open:

```text
Ctrl + Shift + P
→ Preferences: Open User Settings (JSON)
```
and add: 
```json
{
    "C_Cpp.default.compilerPath": "C:/path/to/msys64/ucrt64/bin/g++.exe"
}
```
### 7. Export the Chronos-2 ONNX model
The ONNX model weights are not stored directly in the repository. Run:
```
models/onnx_chronos_v2.ipynb
```
to export the Chronos-2 model.

After export, the models directory must contain:
```
models/
├── chronos2.onnx
└── chronos2.onnx.data
```
Both files are required.
```chronos2.onnx``` contains the ONNX computational graph, while ```chronos2.onnx.data``` contains the external model parameters used by ONNX Runtime.

If ```chronos2.onnx.data``` is absent, ONNX Runtime will fail with an external-data-path error.

### 8. Build and run
Open PowerShell in the repository root and run:

```bash
.\build.bat
```

The build script performs: 
```
CMake configuration
        ↓
Ninja compilation
        ↓
Bloomberg DLL deployment
        ↓
ONNX Runtime DLL deployment
        ↓
chronos2.exe
```
The generated executable is located at:

```
build\chronos2.exe
```
A successful run should produce output similar to:
```
XAU Curncy         7806 rows
XAG Curncy         7804 rows
...
TOTAL             69465

[split] train [...] purged 21 test [...]
[chronos2] quantiles=21 horizon=21 median_idx=10
```

## Appendix: `build.bat`

The repository uses a small Windows batch script to configure, compile, deploy the required runtime libraries, and execute the project.

Before use, update the `UCRT64` and `BLPAPI_ROOT` variables to match the local installation paths.

```bat
@echo off
setlocal

cd /d "%~dp0"

set "UCRT64=C:\Users\userName\Downloads\c inst\ucrt64"

set "BLPAPI_ROOT=C:\Users\userName\Downloads\blpapi_cpp_3.26.7.1-windows\blpapi_cpp_3.26.7.1"

set "PATH=%UCRT64%\bin;%PATH%"

if exist build (
    rmdir /s /q build
)

echo === Paths ===
echo UCRT64      = %UCRT64%
echo BLPAPI_ROOT = %BLPAPI_ROOT%
echo.

echo === Configuring ===

cmake -S . -B build ^
    -G Ninja ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ^
    -DUSE_BLPAPI=ON ^
    "-DONNXRUNTIME_ROOT=%UCRT64%" ^
    "-DBLPAPI_ROOT=%BLPAPI_ROOT%"

if errorlevel 1 goto :error

echo.
echo === Building ===

cmake --build build

if errorlevel 1 goto :error

echo.
echo === Copying runtime DLLs ===

copy /Y "%BLPAPI_ROOT%\lib\blpapi3_64.dll" "build\blpapi3_64.dll" >nul
copy /Y "%UCRT64%\bin\onnxruntime.dll" "build\onnxruntime.dll" >nul
copy /Y "%UCRT64%\bin\onnxruntime_providers_shared.dll" "build\onnxruntime_providers_shared.dll" >nul

echo.
echo === Running chronos2 ===
echo.

build\chronos2.exe

if errorlevel 1 goto :runerror

echo.
echo === Finished successfully ===

pause
exit /b 0

:error
echo.
echo === BUILD FAILED ===
pause
exit /b 1

:runerror
echo.
echo === PROGRAM EXITED WITH AN ERROR ===
pause
exit /b 1
```

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

The database contains 69,465 daily observations across nine Bloomberg instruments. Evaluation was conducted on `XAU Curncy`, using 7,806 observations and a purged chronological test set of 2,321 forecast origins. Each model forecasts 21-day realised volatility; lower QLIKE indicates better predictive accuracy.

| Model                              |      QLIKE | Skill vs HAR-RV |
| ---------------------------------- | ---------: | --------------: |
| TCN (trained from scratch)         | **0.2441** |     **+11.54%** |
| HAR-RV                             |     0.2760 |               — |
| Chronos-2 zero-shot (last of path) |     0.2787 |          −0.99% |
| Chronos-2 zero-shot (mean of path) |     0.2912 |          −5.52% |
| Persistence                        |     0.3308 |         −19.87% |

The TCN achieved the lowest QLIKE, improving upon HAR-RV by 11.54%. However, this improvement was not statistically significant under the Diebold–Mariano test (`DM = −1.48`, `p = 0.140`). Its advantage over zero-shot Chronos-2 was likewise not significant (`p = 0.235`), although it significantly outperformed persistence at the 5% level (`p = 0.042`).

Zero-shot Chronos-2 using the terminal forecast was statistically indistinguishable from HAR-RV (`DM = 0.13`, `p = 0.896`), despite receiving no task-specific training. The mean-of-path Chronos-2 forecast also significantly outperformed persistence (`DM = −4.04`, `p < 0.001`). Overall, the TCN produced the strongest point estimate, while the statistical tests did not establish a significant difference in predictive accuracy among the TCN, HAR-RV and zero-shot Chronos-2.


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
- fractional differencing is written although not yet applied on the data
