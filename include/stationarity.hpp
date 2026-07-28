#pragma once

#include <eigen3/Eigen/Dense>

#include <utility>
#include <vector>

/*
┌───────────────────────────────────────────────────┐
│   Augmenetd Dickey-Fuller (ADF)                   │
└───────────────────────────────────────────────────┘
*/

class Adfuller {
public: 
    Eigen::MatrixXd A; 
    Eigen::VectorXd B;

    std::vector<double> x; 

    std::vector<double> diff();

    /*                      Ax = b

    Where A and b are matrices (b could be a vector, as a special case). 
    You want to find a solution x.
    */

    void buildRegression(int k, int t0);

    std::pair<int, Eigen::VectorXd> selectLagAIC(int k_max);

    double tau(const Eigen::MatrixXd& A, const Eigen::VectorXd& B, const Eigen::VectorXd& beta);

    double mackinnonp_c(double tau);
};



/*
┌───────────────────────────────────────────────────┐
│   Fractional Diff. (de Prado)                     │
└───────────────────────────────────────────────────┘
*/

// FracDiff runs on train data only -- don't forget about train-test split! 15/07/2026

class FracDiff {
public:
    // data members      
    double tau = 1e-4;
    double alpha = 0.05; 
    std::vector<double> w {1.0};

    // input values
    std::vector<double> x;

    // member functions 
    std::vector<double> computeWeights(double d);

    std::vector<double> applyWeights(const std::vector<double>& series, const std::vector<double>& w);

    double gridSearch();
};