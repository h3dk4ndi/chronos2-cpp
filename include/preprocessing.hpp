#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "types.hpp"
#include "sqlite_storage.hpp"

/*
┌───────────────────────────────────────────────────┐
│   Preprocessing (Feature Engineering)             │
└───────────────────────────────────────────────────┘
*/

class Preprocessing {
public: 
    InstrumentMeta meta;

    std::vector<int64_t> dates;
    std::vector<double> open; 
    std::vector<double> high; 
    std::vector<double> low; 
    std::vector<double> close; 
    int roll_w; 

    Preprocessing(SQLite& sql, const std::string& sec, int window);

    // member functions 

    // * Rolling Sum (a substitute to the Pandas .rolling() and .sum() combined)
    std::vector<double> Roll_sum(const std::vector<double>& input, size_t roll_w);


    // * Rolling Mean (a substitute to the Pandas .rolling() and .mean() combinded)
    std::vector<double> Roll_mean(const std::vector<double>& input, size_t roll_w);


    // * Rolling Variance (a substitute to the Pandas .rolling() and .var() combinded)
    std::vector<double> Roll_var(const std::vector<double>& input, size_t roll_w);


    // Return  
    std::vector<double> Returns();


    // Group T: Target Family 

    // Close-to-close Realised Volatility 
    std::vector<double> Close2CloseRV();


    // Parkinson 
    std::vector<double> Parkinson();


    // Garman Klass
    std::vector<double> GarmanKlass();


    // Rogers Satchell
    std::vector<double> RogersSatchell();


    // Yang Zhang 
    std::vector<double> YangZhang();


    // Negative and positive realised semivariance
    std::pair<std::vector<double>, std::vector<double>> Realised_semivar();

    // Signed Jump 
    std::vector<double> Signed_jump(); 

    // Leverage Lag
    std::pair<std::vector<double>, std::vector<double>> Leverage_lag();
    
    // Bipower variation
    std::vector<double> Bipower_variation(); 

    // Jump component
    std::vector<double> Jump_component();
    
    // Relative Jump 
    std::vector<double> Relative_jump();

    // Model Target 
    std::vector<double> ModelTarget();

    // 19/07/2026 22:06 - Add more!

    // 20/07/2026 22:30 - Store preprocessed values in SQL 
};