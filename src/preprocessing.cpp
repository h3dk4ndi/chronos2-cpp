#include "preprocessing.hpp"

#include <cmath>
#include <limits>
#include <numeric>
#include <utility>
#include <algorithm>

Preprocessing::Preprocessing(SQLite& sql, const std::string& sec, int window)
    : meta(sql.loadMeta(sec)),
    roll_w(window)
{
    SQLite::MarketData m = sql.loadBLP(sec);

    dates = std::move(m.dates);
    open  = std::move(m.open);
    high  = std::move(m.high);
    low   = std::move(m.low);
    close = std::move(m.close);
}

std::vector<double> Preprocessing::Roll_sum(const std::vector<double>& input, size_t roll_w) {
    std::vector<double> output(input.size(), std::numeric_limits<double>::quiet_NaN());

    for (size_t i = roll_w - 1; i < input.size(); ++i) {
        output[i] = std::accumulate(
            input.begin() + (i - roll_w + 1),
            input.begin() + (i + 1),
            0.0
        );
    }
    return output;
}

std::vector<double> Preprocessing::Roll_mean(const std::vector<double>& input, size_t roll_w) {
    std::vector<double> output = Roll_sum(input, roll_w);
    for (double& v : output) v /= (double)roll_w;
    return output;
}

std::vector<double> Preprocessing::Roll_var(const std::vector<double>& input, size_t roll_w) {
    std::vector<double> output(input.size(), std::numeric_limits<double>::quiet_NaN());

    for (size_t i = roll_w - 1; i < input.size(); ++i) {
        // Check thoroughly !!!
        auto start = input.begin() + (i - roll_w + 1);
        auto end   = input.begin() + (i + 1);

        double mean =
            std::accumulate(start, end, 0.0) / roll_w;

        double squared_diff = 0.0;

        for (auto it = start; it != end; ++it) {
            squared_diff += (*it - mean) * (*it - mean);
        }

        output[i] = squared_diff / (roll_w - 1);
    }
    return output;
}

std::vector<double> Preprocessing::Returns() {
    std::vector<double> output;

    if (meta.method == "log") {
        for (size_t i = 1; i < close.size(); ++i) {
            output.push_back(std::log(close[i]/close[i - 1])); 
        }
    }
    else if (meta.method == "diff"){
        for (size_t i = 1; i < close.size(); ++i) {
            output.push_back(close[i] - close[i - 1]);
        }
    }
    else if (meta.method == "pct") {
        for (size_t i = 1; i < close.size(); ++i) {
            output.push_back((close[i] - close[i - 1]) / close[i - 1]);
        }
    }
    return output;
}

std::vector<double> Preprocessing::Close2CloseRV() {
    std::vector<double> r = Returns();
    std::vector<double> r_sq;

    r_sq.reserve(r.size());     // ???

    for (double value : r) {
        r_sq.push_back(value * value);
    }

    std::vector<double> output = Roll_sum(r_sq, roll_w);
    const double scale = 252.0 / static_cast<double>(roll_w);

    for (double& value : output) {
        value *= scale;
    }

    return output;
}

std::vector<double> Preprocessing::Parkinson() {
    std::vector<double> hl_sq;
    for (size_t i = 0; i < close.size(); ++i) {
        hl_sq.push_back(std::log(high[i] / low[i]) * std::log(high[i] / low[i])); 
    } 
    std::vector<double> output = Roll_sum(hl_sq, roll_w);

    double scale = 252.0 / (roll_w * 4.0 * std::log(2.0));

    for (double& value : output) {
        value *= scale;
    }
    return output;
}

std::vector<double> Preprocessing::GarmanKlass() { 
    std::vector<double> term; 

    for (size_t i = 0; i < close.size(); ++i) {
        term.push_back(0.5 * std::log(high[i] / low[i]) * std::log(high[i] / low[i]) 
        - ((2 * std::log(2) - 1) * std::log(close[i] / open[i]) * std::log(close[i] / open[i])));
    }
    std::vector<double> output = Roll_sum(term, roll_w);

    double scale = 252.0 / roll_w;

    for (double& value : output) {
        value *= scale;
    }
    return output;
}

std::vector<double> Preprocessing::RogersSatchell() {
    std::vector<double> term;
    for (size_t i = 0; i < close.size(); ++i) {
        term.push_back(std::log(high[i] / close[i]) * std::log(high[i] / open[i]) 
        + std::log(low[i] / close[i]) * std::log(low[i] / open[i])); 
    }
    std::vector<double> output = Roll_sum(term, roll_w);

    double scale = 252.0 / roll_w;

    for (double& value : output) {
        value *= scale;
    }
    return output;
}

std::vector<double> Preprocessing::YangZhang() {
    std::vector<double> overnight; 
    std::vector<double> open_close; 
    std::vector<double>  rs = RogersSatchell(); 

    for (size_t i = 1; i < close.size(); ++i) {
        overnight.push_back(std::log(open[i] / close[i - 1])); 
        open_close.push_back(std::log(close[i] / open[i])); 
    }
    std::vector<double> overnight_var = Roll_var(overnight, roll_w);
    std::vector<double> open_close_var = Roll_var(open_close, roll_w);

    const double k = 0.34 / (1.34 + (static_cast<double>(roll_w + 1) / static_cast<double>(roll_w - 1)));

    std::vector<double> output(overnight.size(), std::numeric_limits<double>::quiet_NaN());

    for (size_t i = 0; i < output.size(); ++i) {
        output[i] =
            252.0 * overnight_var[i]
            + 252.0 * k * open_close_var[i]
            + (1.0 - k) * rs[i + 1];
    }

    return output;
}

// check !
std::pair<std::vector<double>, std::vector<double>> Preprocessing::Realised_semivar() {
    const std::vector<double> r = Returns();

    std::vector<double> r_sq_neg;
    std::vector<double> r_sq_pos;

    r_sq_neg.reserve(r.size());
    r_sq_pos.reserve(r.size());

    for (double value : r) {
        const double squared = value * value;

        // Equivalent to r2 * (r < 0)
        r_sq_neg.push_back(value < 0.0 ? squared : 0.0);

        // Equivalent to r2 * (r > 0)
        r_sq_pos.push_back(value > 0.0 ? squared : 0.0);
    }

    std::vector<double> output_neg = Roll_sum(r_sq_neg, roll_w);
    std::vector<double> output_pos = Roll_sum(r_sq_pos, roll_w);

    const double scale =
        252.0 / static_cast<double>(roll_w);

    for (std::size_t i = 0; i < r.size(); ++i) {
        output_neg[i] *= scale;
        output_pos[i] *= scale;
    }

    return {
        std::move(output_neg),
        std::move(output_pos)
    };
}

// 18/08/2026 -- needs thorough review
// ---- Group A ----
std::vector<double> Preprocessing::Signed_jump() {
    auto rsv = Realised_semivar();                       // {neg, pos}
    std::vector<double> out(rsv.first.size());
    for (size_t i = 0; i < out.size(); ++i)
        out[i] = rsv.second[i] - rsv.first[i];           // pos - neg
    return out;
}

std::pair<std::vector<double>, std::vector<double>> Preprocessing::Leverage_lag() {
    const std::vector<double> r = Returns();
    std::vector<double> lev(r.size());
    for (size_t i = 0; i < r.size(); ++i) lev[i] = std::min(r[i], 0.0);
    return { lev, Roll_mean(lev, 5) };                   // 5, NOT roll_w
}

// ---- Group J ----
std::vector<double> Preprocessing::Bipower_variation() {
    const std::vector<double> r = Returns();
    std::vector<double> prod(r.size(), std::numeric_limits<double>::quiet_NaN());
    for (size_t i = 1; i < r.size(); ++i)                // prod[0] stays NaN
        prod[i] = std::abs(r[i]) * std::abs(r[i - 1]);
    std::vector<double> out = Roll_sum(prod, roll_w);
    const double scale = (252.0 / (double)roll_w) * (M_PI / 2.0);
    for (double& v : out) v *= scale;
    return out;
}

std::vector<double> Preprocessing::Jump_component() {
    const std::vector<double> rv = Close2CloseRV();
    const std::vector<double> bv = Bipower_variation();
    std::vector<double> out(rv.size(), std::numeric_limits<double>::quiet_NaN());
    for (size_t i = 0; i < rv.size(); ++i)
        if (std::isfinite(rv[i]) && std::isfinite(bv[i]))
            out[i] = std::max(rv[i] - bv[i], 0.0);
    return out;
}

std::vector<double> Preprocessing::Relative_jump() {
    const std::vector<double> rv   = Close2CloseRV();
    const std::vector<double> jump = Jump_component();
    std::vector<double> out(rv.size(), std::numeric_limits<double>::quiet_NaN());
    for (size_t i = 0; i < rv.size(); ++i)
        if (std::isfinite(jump[i]) && std::isfinite(rv[i]) && rv[i] > 0.0)
            out[i] = jump[i] / rv[i];
    return out;
}


std::vector<double> Preprocessing::ModelTarget() {
    std::vector<double> rv = Close2CloseRV();                 
    const size_t n = rv.size();
    std::vector<double> y(n, std::numeric_limits<double>::quiet_NaN());

    for (size_t i = 0; i + roll_w < n; ++i)                   // stop where the source runs out
        y[i] = std::log(rv[i + roll_w]);                      // shift(-roll_w)

    return y;                                                 
}