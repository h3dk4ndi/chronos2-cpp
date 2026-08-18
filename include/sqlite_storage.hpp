#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "types.hpp"

struct sqlite3;          // forward-declared: sqlite3.h stays in the .cpp
struct sqlite3_stmt;

/*
┌───────────────────────────────────────────────────┐
│   SQLite - data storage                           │
└───────────────────────────────────────────────────┘
*/

/*
int sqlite3_exec(
  sqlite3*,                                  // An open database 
  const char *sql,                           // SQL to be evaluated 
  int (*callback)(void*,int,char**,char**),  // Callback function 
  void *,                                    // 1st argument to callback 
  char **errmsg                              // Error msg written here 
);
*/

// MY ATTEMPT 
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
    std::vector<double> returns, close2closeRV, parkinson, garmanKlass,
                        rogersSatchell, yangZhang,
                        negativeRealisedSemivar, positiveRealisedSemivar,
                        bipowerVariation, signedJump, leverage,
                        leverageMean5, jumpComponent, relativeJump,
                        modelTarget;
    };


    // member functions 
    void begin();
    void commit();

    // Open the database
    sqlite3*  openDB(const char* filename);

    // Close the database
    bool closeDB();
    // Create Table -> Insert Values -> Read Values (i.e. convertable to c_str())

    // Bloomberg Data Handler
    void CreateBLP();

    void InsertBLP(
        const std::string& security, const std::string& date,
        const std::string& field, double value
    );

    // 22/07/26 08:41 - read SQL, return std::vector<double>
    MarketData loadBLP(const std::string& sec);

    // Instrument_meta (for data classification) Handler 
    void createMetaTable();

    // Assigns the LOG, DIFF, or PCT to each ticker
    void Check();

    void insertMeta(const std::string& security, const std::string& security_typ, const std::string& security_typ2,
                    const std::string& market_sector, const std::string& name,
                    const std::string& currency, const std::string& method);

    // InstrumentMeta is defined after #include
    InstrumentMeta loadMeta(const std::string& sec);

        // Preprocessed Data Handler
    void dropPrep();          // one-shot: drop prep_data so createPrep can rebuild the schema
    void createPrep();

    // Max attention on this one !
    void insertPrep(const std::string& security, const PrepData& d);

    // ! ! !
    PrepData loadPrep(const std::string& sec);


    // Train-Test Split 
    

    // Destructor
    ~SQLite();
};