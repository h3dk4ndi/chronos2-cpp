#include "stationarity.hpp"

#include <math.h>          // M_PI - not guaranteed by <cmath>
#include <cmath>
#include <algorithm>
#include <limits>
#include <stdexcept>

std::vector<double> Adfuller::diff() {
    std::vector<double> delta_x {};
    for (size_t i = 1;i < x.size() ;++i) {
        delta_x.push_back(x[i] - x[i -1]);
    }
    return delta_x;
}

void Adfuller::buildRegression(int k, int t0) {
    std::vector<double> delta_x = diff();

    const int rows = static_cast<int>(x.size()) - t0 - 1; 
    const int cols = k + 2; 

    A.resize(rows, cols); 
    B.resize(rows); 

    for (int i = 0; i < rows; ++i) {
        const int t = t0 + i; 

        B(i) = delta_x[t];
        A(i, 0) = x[t]; 

        for (int j = 0; j < k; ++j) {
            A(i, 1 + j) = delta_x[t - 1 -j];
        }

        A(i, cols - 1) = 1.0;
    
    }
}

std::pair<int, Eigen::VectorXd> Adfuller::selectLagAIC(int k_max) {
    double bestAIC = std::numeric_limits<double>::infinity(); 
    int bestK = 0;

    for (int k = 0; k <= k_max; ++k) {
        buildRegression(k, k_max); 
    
        Eigen::VectorXd beta = A.colPivHouseholderQr().solve(B); 
        double ssr = (B - A * beta).squaredNorm(); 
        const int n = static_cast<int>(A.rows()); 

        double llf = -0.5 * n * (std::log(2.0 * M_PI) + std::log(ssr/n) + 1.0); 
        double aic = -2.0 * llf + 2.0 * (k + 2); 

        if (aic < bestAIC) {
            bestAIC = aic; 
            bestK = k; 
        }
    }
    
    buildRegression(bestK, bestK); 
    Eigen::VectorXd beta = A.colPivHouseholderQr().solve(B);
    return {bestK, beta}; 
}

double Adfuller::tau(const Eigen::MatrixXd& A, const Eigen::VectorXd& B, const Eigen::VectorXd& beta) {
    double ssr = (B - A * beta).squaredNorm(); 
    double s2 = ssr / (A.rows() - A.cols()); 
    double inv00 = (A.transpose() * A).ldlt().solve(Eigen::VectorXd::Unit(A.cols(), 0))(0);
    double se = std::sqrt(s2 * inv00);

    return beta(0) / se;
}

double Adfuller::mackinnonp_c(double tau) {
    const double tau_star = -1.61, tau_min = -18.83, tau_max = 2.74;
    if (tau > tau_max) return 1.0;
    if (tau < tau_min) return 0.0;

    std::vector<double> coef;
    if (tau <= tau_star)
        coef = {2.1659, 1.4412, 0.038269};              // 3 coeffs
    else
        coef = {1.7339, 0.93202, -0.12745, -0.010368};  // 4 coeffs

    double z = 0.0;                                       // Horner, length-agnostic
    for (int i = static_cast<int>(coef.size()) - 1; i >= 0; --i)
        z = z * tau + coef[i];

    return 0.5 * std::erfc(-z / std::sqrt(2.0));          // Φ(z)
}

std::vector<double> FracDiff::computeWeights(double d) {
    w = {1.0}; 
    for (size_t k = 1; ;++k) {
        double w_new = -w.back() * (d - k + 1)/k; 
        if (std::fabs(w_new) < tau) break;
        w.push_back(w_new);
    }
    return w;
}

std::vector<double> FracDiff::applyWeights(const std::vector<double>& series, const std::vector<double>& w) {
    std::vector<double> out; 
    // window width 
    const int W = static_cast<int>(w.size() - 1); 

    for (int t = W; t < static_cast<int>(series.size()); ++t) {
        double acc = 0.0; 

        for (int k = 0; k < static_cast<int>(w.size()); ++k) {
            acc += w[k] * series[t - k];
        }
        out.push_back(acc);
    } 
    return out;
}

double FracDiff::gridSearch() {
    for (double d = 0.0; d <= 1.0 + 1e-9; d += 0.05) {
        std::vector<double> w = computeWeights(d); 
        std::vector<double> xfd = applyWeights(x, w); 

        Adfuller adf; 
        adf.x = xfd; 

        // why static_cast? Note to myself in the future 15/07/2026 21:11
        int n = static_cast<int>(xfd.size());
        if (n < 20) continue;           // ???
        
        int k_max = static_cast<int>(std::ceil(12.0 * std::pow(n / 100.0, 0.25)));
        k_max = std::min(n / 2 - 2, k_max); 

        auto [k, beta] = adf.selectLagAIC(k_max); 
        double t = adf.tau(adf.A, adf.B, beta); 
        double p = adf.mackinnonp_c(t); 

        if (p < alpha) return d; 
    }
    throw std::runtime_error("no d in [0,1] achieved stationarity");
}