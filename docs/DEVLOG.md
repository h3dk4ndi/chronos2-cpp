# Development log

```
09/07/2026 09:13
TODO: Rewrite the code (SQLite + Bloomberg) using struct/classes
└ 1st ✅ 09/07/2026 13:56 | rewritten Bloomberg runs
└ 2nd ✅ 10/07/2026 15:24 | SQlite is running smoothly

11/07/2026 08:02
TODO: Make Bloomberg & SQLite to handle multiple securities and fields in one database
└ 1st ✅ 11/07/2026 09:05 | Multiple security & field support added
└ 2nd ✅ 11/07/2026 09:30 | All data drawn & stored in SQLite

TODO: Start rewriting Preprocessing part from Chronos-2
└ 1st ✅ 15/07/2026 18:33 | Augmented Dickey-Fuller added
└ 2nd ✅ 15/07/2026 22:25 | FracDiff completed

TODO: Non-mapping solution for data preprocessing classification
The idea is to used in-built semantics to automatically classify an asset 
└     ✅ 18/07/2026 00:41 | Added
16/07/2026 00:52
TODO: Bloomberg //blp/refdata - ReferenceRequestData for SECURITY_TYP, etc
└     ✅ 16/07/2026 04:07 | Added
└     ✅ 17/07/2026 23:44 | No hard-coded selection 
   -> Check() function instead (assigns 'log', 'diff', or 'pct' to 'instrument_meta')

TODO: Preprocessing - Main Mechanism (Garman-Klass, Rogers-Satchell, etc)

20/07/2026 04:42 
TODO: Store the Preprocessed values in SQL table for further use in Chronos-2 

TODO: pybind11 for Chronos-2 (19/07/2026 22:06 - switch to ONNX)
└     ✅ 22/07/2026 14:54 | Added, but needs tests for date - value alignment 

21/07/2026 00:06 
TODO: Simplify the SQL storage system, make it less 'clunky'
└     ✅ 22/07/2026 14:55 | Done

22/07/2026 16:03
TODO: add HAR-RV for the target variable 
```