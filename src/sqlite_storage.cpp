#include "sqlite_storage.hpp"

#include <sqlite3.h>

#include <cstdio>
#include <cmath>
#include <limits>
#include <iostream>

class SQLite {
public:

    // data members
    sqlite3 *db;
    sqlite3_stmt *stmt;
    //const std::string& sql; 
    char *errMsg = nullptr;

    struct MarketData {
    std::vector<int64_t> dates;
    std::vector<double> open, high, low, close;
    };

    struct PrepData {
        std::vector<int64_t> dates;
        std::vector<double> returns, close2closeRV, parkinson, garmanKlass, rogersSatchell, yangZhang, modelTarget;
    };


    // member functions 

    void SQLite::begin()  { sqlite3_exec(db, "BEGIN",  NULL, 0, &errMsg); }
    void SQLite::commit() { sqlite3_exec(db, "COMMIT", NULL, 0, &errMsg); }

    // Open the database
    sqlite3*  SQLite::openDB(const char* filename) {
        int status = sqlite3_open(filename, &db); 
        if (status != SQLITE_OK) {
            sqlite3_close(db); 
            return nullptr;
        }
        return db;
    }

    // Close the database
    bool SQLite::closeDB() {
        return sqlite3_close(db) == SQLITE_OK;
    }
    // Create Table -> Insert Values -> Read Values (i.e. convertable to c_str())

    // Bloomberg Data Handler
    void SQLite::CreateBLP() {
        const char *sql = R"(
        CREATE TABLE IF NOT EXISTS blp_data (
                date     INTEGER NOT NULL,
                security TEXT NOT NULL,
                field    TEXT NOT NULL,
                value    FLOAT NOT NULL,

                PRIMARY KEY (security, date, field)
            );
        )";
        
        if (sqlite3_exec(db, sql, 0, 0, &errMsg) != SQLITE_OK) {
            printf("SQL error: %s\n", errMsg);
            sqlite3_free(errMsg);
        }
    };


    void SQLite::InsertBLP(
        const std::string& security, const std::string& date,
        const std::string& field, double value
    ) {
        const char *sql = R"(INSERT OR REPLACE INTO blp_Data (date, security, field, value) VALUES (?, ?, ?, ?))";

        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            printf("Failed to prepare statement: %s\n", sqlite3_errmsg(db)); 
        }

        // 21/07/2026 04:18 date should be DATE in SQL 
        sqlite3_bind_int(stmt, 1, std::stoi(date));           
        sqlite3_bind_text(stmt, 2, security.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, field.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_double(stmt, 4, value);

        if (sqlite3_step(stmt) != SQLITE_DONE) {
            printf("Execution failed: %s\n", sqlite3_errmsg(db));
        } 
        sqlite3_finalize(stmt);  // Finalize the prepared statement
    };


    // 22/07/26 08:41 - read SQL, return std::vector<double>
    SQLite::MarketData SQLite::loadBLP(const std::string& sec) {
        MarketData out; 

        const char* sql = R"(
        SELECT date,
            MAX(CASE WHEN field='PX_OPEN' THEN value END),
            MAX(CASE WHEN field='PX_HIGH' THEN value END),
            MAX(CASE WHEN field='PX_LOW'  THEN value END),
            MAX(CASE WHEN field='PX_LAST' THEN value END)
        FROM blp_data
        WHERE security = ?
        GROUP BY date
        HAVING COUNT(DISTINCT field) = 4
        ORDER BY date;
        )"; 

        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            printf("Failed to prepare statement: %s\n", sqlite3_errmsg(db)); 
            return out;
        }

        // read data from SQL, output std::vector<double>
        sqlite3_bind_text(stmt, 1, sec.c_str(), -1, SQLITE_TRANSIENT);
        while (sqlite3_step(stmt) == SQLITE_ROW) {
            out.dates.push_back(sqlite3_column_int64(stmt, 0));
            out.open .push_back(sqlite3_column_double(stmt, 1));
            out.high .push_back(sqlite3_column_double(stmt, 2));
            out.low  .push_back(sqlite3_column_double(stmt, 3));
            out.close.push_back(sqlite3_column_double(stmt, 4));
        }
        sqlite3_finalize(stmt);

        return out;
    };


    // Instrument_meta (for data classification) Handler 
    void SQLite::createMetaTable() {
        const char* sql = R"(
            CREATE TABLE IF NOT EXISTS instrument_meta (
                security        TEXT PRIMARY KEY,
                security_typ    TEXT,
                security_typ2   TEXT,
                market_sector   TEXT,
                name            TEXT,
                currency        TEXT, 
                method          TEXT);
        )";
        if (sqlite3_exec(db, sql, NULL, 0, &errMsg) != SQLITE_OK) {
            printf("Error: %s \n", errMsg); sqlite3_free(errMsg);
        }
    };


    // Assigns the LOG, DIFF, or PCT to each ticker
    void Check() {
        const char* sql = R"(
            UPDATE instrument_meta
            SET method =
                CASE
                    -- Series contains zero or negative observations
                    WHEN EXISTS (
                        SELECT 1
                        FROM blp_data
                        WHERE blp_data.security = instrument_meta.security
                        AND blp_data.value <= 0.0
                    )
                    THEN 'pct'

                    -- Pure yield/rate index
                    WHEN LOWER(TRIM(security_typ))  = 'index'
                    AND LOWER(TRIM(security_typ2)) = 'index'
                    THEN 'diff'

                    -- Positive price-like series
                    ELSE 'log'
                END;
        )";

        if (sqlite3_exec(db, sql, nullptr, nullptr, &errMsg) != SQLITE_OK) {
            std::cerr << "Check failed: " << errMsg << '\n';
            sqlite3_free(errMsg);
            errMsg = nullptr;
        }
    };


    void SQLite::insertMeta(const std::string& security, const std::string& security_typ, const std::string& security_typ2,
                    const std::string& market_sector, const std::string& name,
                    const std::string& currency, const std::string& method) {
        if (sqlite3_prepare_v2(db,
            "INSERT OR REPLACE INTO instrument_meta VALUES (?,?,?,?,?,?,?)", -1, &stmt, nullptr) != SQLITE_OK) {
            printf("insertMeta prepare failed: %s\n", sqlite3_errmsg(db));
            return;
        }
        sqlite3_bind_text(stmt, 1, security.c_str(),      -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 2, security_typ.c_str(),  -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 3, security_typ2.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 4, market_sector.c_str(), -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 5, name.c_str(),          -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 6, currency.c_str(),      -1, SQLITE_TRANSIENT);
        sqlite3_bind_text(stmt, 7, method.c_str(),      -1, SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_finalize(stmt);
    };


    // InstrumentMeta is defined after #include
    InstrumentMeta SQLite::loadMeta(const std::string& sec) {
        InstrumentMeta m;
        sqlite3_prepare_v2(db,
            "SELECT security_typ, market_sector, currency, method FROM instrument_meta WHERE security=?",
            -1, &stmt, nullptr);
        sqlite3_bind_text(stmt, 1, sec.c_str(), -1, SQLITE_TRANSIENT);

        // 22/07/2026 09:04 - purpose of 'reinterpret_cast'?
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            m.securityType  = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0));
            m.marketSector  = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
            m.currency      = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 2));
            m.method        = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 3));
        }
        sqlite3_finalize(stmt);
        return m;
    };


    // Preprocessed Data Handler
    void SQLite::createPrep() {
        const char* sql = R"(
            CREATE TABLE IF NOT EXISTS prep_data(
                date            INTEGER NOT NULL,
                security        TEXT    NOT NULL,
                returns         REAL,
                close2closeRV   REAL,
                parkinson       REAL,
                garmanKlass     REAL,
                rogersSatchell  REAL,
                yangZhang       REAL,
                modelTarget     REAL,
                PRIMARY KEY (security, date)
            );
        )";
        if (sqlite3_exec(db, sql, NULL, 0, &errMsg) != SQLITE_OK) {
            printf("createPrep error: %s \n", errMsg); sqlite3_free(errMsg);
        }
    };


    // Max attention on this one !
    void SQLite::insertPrep(
        const std::string& security, const std::vector<int64_t>& dates,
        const std::vector<double>& returns, const std::vector<double>& close2closeRV,
        const std::vector<double>& parkinson, const std::vector<double>& garmanKlass,
        const std::vector<double>& rogersSatchell, const std::vector<double>& yangZhang, 
        const std::vector<double>& modelTarget
    ) {
        if (sqlite3_prepare_v2(db,
            "INSERT OR REPLACE INTO prep_data "
            "(date, security, returns, close2closeRV, parkinson, garmanKlass, rogersSatchell, yangZhang, modelTarget) "
            "VALUES (?,?,?,?,?,?,?,?,?)", -1, &stmt, nullptr) != SQLITE_OK) {
            printf("insertPrep prepare failed: %s\n", sqlite3_errmsg(db));   // was: "insertMeta" — fix the label
            return;
        }

        auto bind = [&](int col, const std::vector<double>& v, size_t i) {
            long long j = (long long)i - (long long)(dates.size() - v.size());   // tail alignment
            if (j >= 0 && std::isfinite(v[j])) {
                sqlite3_bind_double(stmt, col, v[j]);
            } else {
                sqlite3_bind_null(stmt, col);
            }
        };

        for (size_t i = 0; i < dates.size(); ++i) {
            sqlite3_bind_int64(stmt, 1, dates[i]);
            sqlite3_bind_text (stmt, 2, security.c_str(), -1, SQLITE_TRANSIENT);
            bind(3, returns, i); 
            bind(4, close2closeRV, i);
            bind(5, parkinson, i);
            bind(6, garmanKlass, i);
            bind(7, rogersSatchell, i);
            bind(8, yangZhang, i);
            bind(9, modelTarget, i);

            if (sqlite3_step(stmt) != SQLITE_DONE)
                printf("insertPrep failed at %lld: %s\n", (long long)dates[i], sqlite3_errmsg(db));
            sqlite3_reset(stmt);            // rewind so the next iteration can bind+step again
        }
        sqlite3_finalize(stmt);
    };

    // limits for loadPrep!

    // ! ! !
    SQLite::PrepData SQLite::loadPrep(const std::string& sec) {
        PrepData out;

        const char* sql = R"(
            SELECT date, returns, close2closeRV, parkinson,
                garmanKlass, rogersSatchell, yangZhang, modelTarget
            FROM prep_data
            WHERE security = ?
            ORDER BY date;
        )";

        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            printf("loadPrep prepare failed: %s\n", sqlite3_errmsg(db));
            return out;
        }
        sqlite3_bind_text(stmt, 1, sec.c_str(), -1, SQLITE_TRANSIENT);

        auto val = [&](int i) {          // NULL -> NaN (same NaN Roll_sum already emits)
            return sqlite3_column_type(stmt, i) == SQLITE_NULL
                ? std::numeric_limits<double>::quiet_NaN()
                : sqlite3_column_double(stmt, i);
        };

        while (sqlite3_step(stmt) == SQLITE_ROW) {
            out.dates         .push_back(sqlite3_column_int64(stmt, 0));
            out.returns       .push_back(val(1));
            out.close2closeRV .push_back(val(2));
            out.parkinson     .push_back(val(3));
            out.garmanKlass   .push_back(val(4));
            out.rogersSatchell.push_back(val(5));
            out.yangZhang     .push_back(val(6));
            out.modelTarget   .push_back(val(7));
        }
        sqlite3_finalize(stmt);
        return out;
    }
    

    // Train-Test Split 
    
    


    // Destructor
    SQLite::~SQLite() { if (db) sqlite3_close(db); }
};