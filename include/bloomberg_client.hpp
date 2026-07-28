#pragma once

#include <blpapi_session.h>
#include <blpapi_service.h>
#include <blpapi_request.h>

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "types.hpp"
#include "sqlite_storage.hpp"

/*
┌───────────────────────────────────────────────────┐
│   Bloomberg - data connectivity and extraction    │
└───────────────────────────────────────────────────┘
*/

class Bloomberg {
public: 
    // data member(s)

    // Connect() 
    blpapi::SessionOptions options; 
    blpapi::Session session{options};

    // DataReq()
    blpapi::Service dataService; 
    std::unique_ptr<blpapi::Request> req;       // !
    std::unique_ptr<blpapi::Request> req_meta;
    // v2.0 std::unique_ptr<Request> req;
    // v1.0 (my attempt) Request req;

    // CoreInputs()
    std::vector<std::string> security; 
    std::vector<std::string> field;
    std::string startDate, endDate;

    // EventLoop2()
    //Datetime date;
    std::string date;

    // MetaData 
    std::vector<std::string> metaField {
        "SECURITY_TYP",
        "SECURITY_TYP2",
        "MARKET_SECTOR_DES",
        "NAME",
        "CRNCY"
    };

    // member on Bloomberg — THIS is where metadata must live to survive the loop:
    std::map<std::string, InstrumentMeta> meta;



    // Constructor 
    Bloomberg(std::vector<std::string> sec, std::vector<std::string> fld, std::string start, std::string end);

    // member function(s)
    bool Connect();

    bool DataReq();

    // Metadata Object - MetaReq()  
    bool MetaReq();

    // why do we use -> here? 
    bool CoreInputs();

    // MetaObject - MetaInputs()
    bool MetaInputs();

    bool ReqOptions();

    bool sendRequest();

    bool EventLoop();

    // 22/07/2026 08:02 - added newer version of SQL class in operation 
    bool EventLoop2(SQLite& sql);

    bool MetaLoop(SQLite& store);

};