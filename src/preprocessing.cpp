#include "preprocessing.hpp"

#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

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

std::vector<double> Preprocessing::ModelTarget() {
    std::vector<double> rv = Close2CloseRV();                 
    const size_t n = rv.size();
    std::vector<double> y(n, std::numeric_limits<double>::quiet_NaN());

    for (size_t i = 0; i + roll_w < n; ++i)                   // stop where the source runs out
        y[i] = std::log(rv[i + roll_w]);                      // shift(-roll_w)

    return y;                                                 
}