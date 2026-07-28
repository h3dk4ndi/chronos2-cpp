#include "bloomberg_client.hpp"

#include <iostream>

using namespace BloombergLP; 
using namespace blpapi; 

// Constructor 
Bloomberg::Bloomberg(std::vector<std::string> sec, std::vector<std::string> fld, std::string start, std::string end)
    : options()
    , session(options)
    , security(sec)
    , field(fld)
    , startDate(start)
    , endDate(end)
{}

// member function(s)
bool Bloomberg::Connect() {
    options.setServerHost("localhost");
    options.setServerPort(8194); 

    if (!session.start()) {
        std::cerr << "Failed to start session. " << '\n';
        return 1;
    }

    if (!session.openService("//blp/refdata")) {
        std::cerr << "Failed to open //blp/refdata" << '\n';
        return 1;
    }
    return true; 
}

bool Bloomberg::DataReq() {
    Service dataService = session.getService("//blp/refdata");
    //req = std::make_unique<Request>(ref.createRequest("HistoricalDataRequest"));

    /* 
    what's the purpose std::make_unique if 
    https://bloomberg.github.io/blpapi-docs/cpp/3.7/classblpapi_1_1Request.html
    defines differently
    */

    req = std::make_unique<blpapi::Request>(dataService.createRequest("HistoricalDataRequest"));
    /*session.getService("//blp/refdata");
    Request req(dataService.createRequest("HistoricalDataRequest"));
    */
    return true;
}

// Metadata Object - MetaReq()  
bool Bloomberg::MetaReq() {
    Service dataService = session.getService("//blp/refdata");
    req_meta = std::make_unique<blpapi::Request>(dataService.createRequest("ReferenceDataRequest"));
    return true; 
}

// why do we use -> here? 
bool Bloomberg::CoreInputs() {
    for (const std::string& s : security) {
        req->getElement(Name("securities")).appendValue(s.c_str());
    }
    
    for (const std::string& f : field) {
       req->getElement(Name("fields")).appendValue(f.c_str()); 
    }

    req->set(Name("startDate"), startDate.c_str());
    req->set(Name("endDate"), endDate.c_str());
    return true;
}

// MetaObject - MetaInputs()
bool Bloomberg::MetaInputs() {
    for (const std::string& s : security) {
        req_meta->getElement(Name("securities")).appendValue(s.c_str());
    }
    
    /*
    for (const std::string& f : field) {
       req_meta->getElement(Name("fields")).appendValue(f.c_str()); 
    }
    */

    for (const std::string& f : metaField) {
        req_meta->getElement(Name("fields")).appendValue(f.c_str());
    }

    return true; 
}

bool Bloomberg::ReqOptions() {
    req->set(Name("periodicitySelection"), "DAILY");
    req->set(Name("periodicityAdjustment"), "ACTUAL");
    req->set(Name("adjustmentNormal"), true);
    req->set(Name("adjustmentAbnormal"), true);
    req->set(Name("adjustmentSplit"), true);
    req->set(Name("adjustmentFollowDPDF"), true);
    req->set(Name("pricingOption"), "PRICING_OPTION_PRICE");
    req->set(Name("overrideOption"), "OVERRIDE_OPTION_CLOSE");
    req->set(Name("nonTradingDayFillOption"), "ACTIVE_DAYS_ONLY");
    return true;
}

bool Bloomberg::sendRequest() {
    session.sendRequest(*req); 
    return true;
}

bool Bloomberg::EventLoop() {
    bool done = false;
    while (!done) {
        Event ev = session.nextEvent();
        for (Message msg : ev) {
            std::cout << msg << std::endl;
            if (ev.eventType() == Event::RESPONSE) {
                done = true;
            }
        }
    }
    return true;
}

// 22/07/2026 08:02 - added newer version of SQL class in operation 
bool Bloomberg::EventLoop2(SQLite& sql) {
    bool done = false; 
    while (!done) {
        Event ev = session.nextEvent(); 
        MessageIterator it(ev);
        while (it.next()) {
            Message msg = it.message();
            
            if (msg.hasElement(Name("securityData"))) {
                Element sd = msg.getElement(Name("securityData")); 
                std::string sec = sd.getElementAsString(Name("security"));
                //std::cout << sd.getElement(Name("fieldData")) << '\n';
                
                if (sd.hasElement(Name("fieldData"))) {
                    Element fd = sd.getElement(Name("fieldData")); 

                    for (size_t i = 0; i< fd.numValues(); ++i) {
                        Element row = fd.getValueAsElement(i);
                        Datetime d = row.getElementAsDatetime(Name("date")); 

                        std::string ds = std::to_string(d.year())
                            + (d.month()<10?"0":"") + std::to_string(d.month())
                            + (d.day()<10?"0":"") + std::to_string(d.day());

                        for (const std::string& f : field) {
                            if (row.hasElement(Name(f.c_str()))) {
                                double px = row.getElementAsFloat64(Name(f.c_str()));
                                //std::cout << ds << " " << sec << "  " << px << std::endl;         enable if needed
                                //store.insertDB(sec, ds, f, px);
                                sql.InsertBLP(sec, ds, f, px);
                            }
                        }
                    }
                }
            }
        }
        if (ev.eventType() == Event::RESPONSE) done = true;
    }
    return false;
}

bool Bloomberg::MetaLoop(SQLite& store) {
    session.sendRequest(*req_meta);
    while (true) {
        Event ev = session.nextEvent();
        MessageIterator it(ev);
        while (it.next()) {
            Message msg = it.message();
            if (msg.messageType() != Name("ReferenceDataResponse")) continue;

            Element arr = msg.getElement(Name("securityData"));
            for (size_t i = 0; i < arr.numValues(); ++i) {
                Element sd = arr.getValueAsElement(i);
                std::string sec = sd.getElementAsString(Name("security"));

                if (sd.hasElement(Name("securityError"))) {
                    std::cerr << "[SECURITY ERROR] " << sec << ": " << sd.getElement(Name("securityError")).getElementAsString(Name("message")) << "\n";
                    continue;
                }

                Element fd = sd.getElement(Name("fieldData"));

                InstrumentMeta m;

                // 18/07/2026 01:03 swap 'if' for something else
                // 21/07/2026 05:41 change 'if' on lambda
                if (fd.hasElement(Name("SECURITY_TYP"))) {
                    m.securityType = fd.getElementAsString(Name("SECURITY_TYP"));
                }
                if (fd.hasElement(Name("SECURITY_TYP2"))) {
                    m.securityType2 = fd.getElementAsString(Name("SECURITY_TYP2"));
                }
                if (fd.hasElement(Name("MARKET_SECTOR_DES"))) {
                    m.marketSector = fd.getElementAsString(Name("MARKET_SECTOR_DES"));
                }
                if (fd.hasElement(Name("NAME"))) {
                    m.name = fd.getElementAsString(Name("NAME"));
                }
                if (fd.hasElement(Name("CRNCY"))) {
                    m.currency     = fd.getElementAsString(Name("CRNCY"));
                }
            
                meta[sec] = m;
                store.insertMeta(sec, m.securityType, m.securityType2, m.marketSector, m.name, m.currency, "");  // <-- ADD THIS
            }
        }
        if (ev.eventType() == Event::RESPONSE) break;
    }
    return true;
}