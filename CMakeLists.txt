cmake_minimum_required(VERSION 3.16)
project(chronos2_cpp CXX)

set(CMAKE_CXX_STANDARD 17)          # structured bindings in FracDiff::gridSearch
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_BLPAPI "Compile the Bloomberg ingest path" OFF)

add_executable(chronos2
    src/main.cpp
    src/sqlite_storage.cpp
    src/stationarity.cpp
    src/preprocessing.cpp
    src/chronos2_onnx.cpp
    src/study.cpp
    src/evaluation.cpp
)

target_include_directories(chronos2 PRIVATE include)

find_package(Eigen3 REQUIRED)
target_link_libraries(chronos2 PRIVATE Eigen3::Eigen sqlite3)

target_include_directories(chronos2 PRIVATE ${ONNXRUNTIME_ROOT}/include)
target_link_directories(chronos2    PRIVATE ${ONNXRUNTIME_ROOT}/lib)
target_link_libraries(chronos2      PRIVATE onnxruntime)

if(USE_BLPAPI)
    target_sources(chronos2 PRIVATE src/bloomberg_client.cpp)
    target_compile_definitions(chronos2 PRIVATE USE_BLPAPI)
    target_include_directories(chronos2 PRIVATE ${BLPAPI_ROOT}/include)
    target_link_directories(chronos2    PRIVATE ${BLPAPI_ROOT}/lib)
    target_link_libraries(chronos2      PRIVATE blpapi3_64)
endif()