#include <cstdio>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>

#include "config.hpp"
#include "types.hpp"
#include "sqlite_storage.hpp"
#include "stationarity.hpp"
#include "split.hpp"
#include "preprocessing.hpp"
#include "chronos2_onnx.hpp"
#include "study.hpp"
#include "evaluation.hpp"

#ifdef USE_BLPAPI
#include "bloomberg_client.hpp"
#endif

/*
┌───────────────────────────────────────────────────┐
│   int main()                                      │
└───────────────────────────────────────────────────┘
*/


int main() {
    const auto t0 = std::chrono::steady_clock::now();

    const bool        REFRESH_DATA = true;          // true only for a fresh pull
    const bool        REBUILD_PREP = true;
    const std::string TARGET_SEC   = "XAU Curncy";

    std::vector<std::string> secs = {
        "XAU Curncy","XAG Curncy","DXY Curncy","EURUSD Curncy","USDJPY Curncy",
        "USGG10YR Index","CL1 Comdty","HG1 Comdty","VIX Index" };

    // ---- 0. database -------------------------------------------------
    SQLite sql;
    if (!sql.openDB("blp.db")) { printf("cannot open blp.db\n"); return 1; }
    sql.CreateBLP();
    sql.createMetaTable();
    if (REBUILD_PREP) sql.dropPrep();
    sql.createPrep();

    // ---- 1. ingest (Terminal required) -------------------------------
#ifdef USE_BLPAPI
    if (REFRESH_DATA) {
        Bloomberg blp1(secs, {"PX_OPEN","PX_HIGH","PX_LOW","PX_LAST"}, "19960601", "20260601");
        blp1.Connect();

        blp1.DataReq(); blp1.CoreInputs(); blp1.ReqOptions(); blp1.sendRequest();
        sql.begin(); blp1.EventLoop2(sql); sql.commit();

        blp1.MetaReq(); blp1.MetaInputs();
        sql.begin(); blp1.MetaLoop(sql); sql.commit();
        sql.Check();
    }
#endif

    // ---- 1b. rebuild features from stored blp_data (no Terminal) -----
    if (REBUILD_PREP) {
        for (size_t i = 0; i < secs.size(); ++i) {
            Preprocessing pp(sql, secs[i], ROLL_W);
            auto rsv = pp.Realised_semivar();
            auto lev = pp.Leverage_lag();

            SQLite::PrepData d;
            d.dates                   = pp.dates;
            d.returns                 = pp.Returns();
            d.close2closeRV           = pp.Close2CloseRV();
            d.parkinson               = pp.Parkinson();
            d.garmanKlass             = pp.GarmanKlass();
            d.rogersSatchell          = pp.RogersSatchell();
            d.yangZhang               = pp.YangZhang();
            d.negativeRealisedSemivar = rsv.first;
            d.positiveRealisedSemivar = rsv.second;
            d.bipowerVariation        = pp.Bipower_variation();
            d.signedJump              = pp.Signed_jump();
            d.leverage                = lev.first;
            d.leverageMean5           = lev.second;
            d.jumpComponent           = pp.Jump_component();
            d.relativeJump            = pp.Relative_jump();
            d.modelTarget             = pp.ModelTarget();

            sql.begin();
            sql.insertPrep(secs[i], d);
            sql.commit();
        }

        size_t total = 0;
        for (size_t i = 0; i < secs.size(); ++i) {
            SQLite::PrepData q = sql.loadPrep(secs[i]);
            total += q.dates.size();
            printf("%-16s %6zu rows\n", secs[i].c_str(), q.dates.size());
        }
        printf("%-16s %6zu (expect 69465)\n", "TOTAL", total);
    }

    // ---- 2. load + build the context matrix --------------------------
    SQLite::PrepData p = sql.loadPrep(TARGET_SEC);
    const size_t N = p.dates.size();
    if (N < (size_t)CONTEXT + ROLL_W + 10) { printf("not enough rows\n"); return 1; }
    printf("\n[data] %s  %zu rows  %lld -> %lld\n", TARGET_SEC.c_str(), N,
           (long long)p.dates.front(), (long long)p.dates.back());

    std::vector<std::vector<double>> M = BuildMatrix(p);
    ReportNaN(M);

    // ---- 3. split (indices only - the table is never truncated) ------
    Split s = TrainTestIndex(N, TEST_FRAC, (size_t)ROLL_W);
    //printf("\n[split] train [0,%zu)  purged %d  test [%zu,%zu)\n",
    //       s.trainEnd, ROLL_W, s.testStart, N);

    size_t first = s.testStart;
    if (first < (size_t)CONTEXT - 1) first = (size_t)CONTEXT - 1;
    const size_t last = N - (size_t)ROLL_W;

    printf("\n[split] train [0,%zu)  purged %d  test [%zu,%zu)  origins %zu..%zu\n",
           s.trainEnd, ROLL_W, s.testStart, N, first, last - 1);

    /*
    ┌───────────────────────────────────────────────────┐
    │ ---- 4. model ----------------------------------- │
    └───────────────────────────────────────────────────┘
    */

    Chronos2ONNX model("models/chronos2.onnx");
    //Chronos2ONNX model("models/chronos2-finetune.onnx");
    if (model.horizon() != ROLL_W) {
        printf("!! graph horizon %lld != ROLL_W %d\n", (long long)model.horizon(), ROLL_W);
        return 1;
    }

    // ---- 5. walk forward ---------------------------------------------
    //size_t first = s.testStart;
    //if (first < (size_t)CONTEXT - 1) first = (size_t)CONTEXT - 1;
    //const size_t last = N - (size_t)ROLL_W;      // modelTarget valid for i < N-w

    std::vector<double> lossC, lossL, lossH, lossP;
    lossC.reserve(2400); lossL.reserve(2400); lossH.reserve(2400); lossP.reserve(2400);
    double beta[4], xrow[4];
    size_t n = 0;

    for (size_t t = first; t < last; ++t) {
        double actual  = p.modelTarget[t];
        double persist = M[0][t];
        if (!std::isfinite(actual) || !std::isfinite(persist)) continue;

        // HAR refits at every origin on an expanding window, purged by ROLL_W -
        // same embargo rule as the split, applied per origin this time.
        if (!HarRow(M[0], t, xrow))                                continue;
        if (!HarFit(M[0], p.modelTarget, 0, t - (size_t)ROLL_W, beta)) continue;
        double har = HarPredict(beta, xrow);

        Chronos2ONNX::Forecast f = model.predict(Window(M, t, (size_t)CONTEXT));
        double mn = 0.0;
        for (size_t i = 0; i < f.median.size(); ++i) mn += f.median[i];
        mn /= (double)f.median.size();      // old convention: mean of the whole path
        double lastv = f.median.back();     // step H -> forecasts log-RV at t+H
        if (!std::isfinite(mn) || !std::isfinite(lastv) || !std::isfinite(har)) continue;

        lossC.push_back(QLike(actual, mn));
        lossL.push_back(QLike(actual, lastv));
        lossH.push_back(QLike(actual, har));
        lossP.push_back(QLike(actual, persist));
        ++n;

        if (n % 200 == 0) {
            double a=0,b=0,c=0,e=0;
            for (size_t i=0;i<n;++i){ a+=lossC[i]; b+=lossL[i]; c+=lossH[i]; e+=lossP[i]; }
            const double el  = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t0).count();
            const double eta = el * ((double)(last - first) / (double)n - 1.0);    
            printf("   %5zu origins  mean %.5f  last %.5f  har %.5f  pers %.5f"
                   "   %.0fs  eta %.0fs\n",
                   n, a/n, b/n, c/n, e/n, el, eta);
        }   
    }
    if (n == 0) { printf("no valid origins\n"); return 1; }


    
    std::vector<double> lossT;
    { FILE* f = fopen("tcn_loss.csv", "r");
      if (f) { double v; while (fscanf(f, "%lf", &v) == 1) lossT.push_back(v); fclose(f); }
      else printf("[tcn] tcn_loss.csv not found -- run run_tcn.py first\n"); }
    

    // ---- 6. report ----------------------------------------------------
    double mc=0, ml=0, mh=0, mp=0;
    for (size_t i=0;i<n;++i){ mc+=lossC[i]; ml+=lossL[i]; mh+=lossH[i]; mp+=lossP[i]; }
    mc/=n; ml/=n; mh/=n; mp/=n;

    DMResult dLM = DieboldMariano(lossL, lossC, (size_t)ROLL_W);   // last vs mean
    DMResult dLH = DieboldMariano(lossL, lossH, (size_t)ROLL_W);   // last vs HAR
    DMResult dCH = DieboldMariano(lossC, lossH, (size_t)ROLL_W);   // mean vs HAR
    DMResult dCP = DieboldMariano(lossC, lossP, (size_t)ROLL_W);

    printf("\n[result] %s   %zu origins\n", TARGET_SEC.c_str(), n);
    printf("   QLIKE chronos MEAN-of-path %.6f\n", mc);
    printf("   QLIKE chronos LAST-of-path %.6f\n", ml);
    printf("   QLIKE HAR-RV               %.6f\n", mh);
    printf("   QLIKE persistence          %.6f\n", mp);
    printf("   skill last vs HAR      %+.2f%%\n", 100.0*(1.0 - ml/mh));
    printf("   DM last vs mean        %+8.4f   p = %.4g\n", dLM.stat, dLM.pvalue);
    printf("   DM last vs HAR         %+8.4f   p = %.4g\n", dLH.stat, dLH.pvalue);
    printf("   DM mean vs HAR         %+8.4f   p = %.4g\n", dCH.stat, dCH.pvalue);
    printf("   DM mean vs persistence %+8.4f   p = %.4g\n", dCP.stat, dCP.pvalue);

    
    if (lossT.size() == n) {
        double mt = 0.0;
        for (size_t i = 0; i < n; ++i) mt += lossT[i];
        mt /= n;

        DMResult dTH = DieboldMariano(lossT, lossH, (size_t)ROLL_W);
        DMResult dTL = DieboldMariano(lossT, lossL, (size_t)ROLL_W);
        DMResult dTP = DieboldMariano(lossT, lossP, (size_t)ROLL_W);

        printf("   QLIKE TCN                  %.6f\n", mt);
        printf("   skill TCN vs HAR       %+.2f%%\n", 100.0*(1.0 - mt/mh));
        printf("   DM TCN vs HAR          %+8.4f   p = %.4g\n", dTH.stat, dTH.pvalue);
        printf("   DM TCN vs chronos last %+8.4f   p = %.4g\n", dTL.stat, dTL.pvalue);
        printf("   DM TCN vs persistence  %+8.4f   p = %.4g\n", dTP.stat, dTP.pvalue);
    } else if (!lossT.empty()) {
        printf("!! tcn origins %zu != %zu -- misaligned, DM skipped\n", lossT.size(), n);
    }
    


    return 0;
}