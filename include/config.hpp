#pragma once

/*
┌───────────────────────────────────────────────────┐
│   Study                                           │
└───────────────────────────────────────────────────┘
*/

inline constexpr int    ROLL_W    = 21;    // must equal the exported graph horizon
inline constexpr int    CONTEXT   = 512;   // graph accepts 64..8192; the study used 256
inline constexpr double TEST_FRAC = 0.30;