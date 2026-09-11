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
          mingw-w64-ucrt-x86_64-onnx
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

The final experiment uses **XAU/USD only** as the forecasting asset. Although additional Bloomberg series were collected during data preparation, the forecasting models use the 14 engineered features derived from XAU/USD rather than cross-asset inputs.

The final evaluation uses a purged train/test split with a 21-day forecast horizon:

```text
[split] train [0,5443)  purged 21  test [5464,7806)
[test origins] 2321
```
Forecast performance:
| Model                   |      QLIKE |
| ----------------------- | ---------: |
| TCN                     | **0.2441** |
| HAR-RV                  |     0.2760 |
| Chronos-2, last-of-path |     0.2787 |
| Chronos-2, mean-of-path |     0.2912 |
| Persistence             |     0.3308 |

The TCN achieved the lowest QLIKE, improving on HAR-RV by approximately **11.5%**.

Diebold-Mariano tests did not show a statistically significant difference between TCN and HAR-RV at the **5%** level ($p = 0.1397$). TCN nevertheless significantly outperformed persistence ($p = 0.0422$).

Chronos-2 substantially outperformed persistence when using the mean forecast path ($p < 0.001$), but did not significantly outperform HAR-RV.

The TCN was trained using:
```text
features = 14
window   = 512
epochs   = 50
batch    = 64
patience = 10
```
Early stopping selected the model at the minimum validation QLIKE:
```
best validation QLIKE = 0.1256
test QLIKE            = 0.2441
```

