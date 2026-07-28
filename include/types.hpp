#pragma once
#include <string>

struct InstrumentMeta {
        std::string securityType;       // SECURITY_TYP
        std::string securityType2;      // SECURITY_TYP2
        std::string marketSector;        // MARKET_SECTOR_DES  (CURNCY/COMDTY/INDEX/GOVT…)
        std::string name;               // NAME
        std::string currency;           // CRNCY
        std::string method;             // METHOD
    };