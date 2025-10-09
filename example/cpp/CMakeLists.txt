cmake_minimum_required(VERSION 3.14)
project(ultralight_dh_onnx_example LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(UL_DH_USE_OPENCV "Link against OpenCV for image IO" ON)

if(NOT DEFINED ONNXRUNTIME_DIR)
    message(FATAL_ERROR "Please set ONNXRUNTIME_DIR to the root of your ONNX Runtime installation.")
endif()

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_DIR}/lib ${ONNXRUNTIME_DIR}/lib64)

if(UL_DH_USE_OPENCV)
    find_package(OpenCV REQUIRED)
endif()

add_executable(ultralight_dh_onnx
    main.cpp
)

target_include_directories(ultralight_dh_onnx PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

if(UL_DH_USE_OPENCV)
    target_link_libraries(ultralight_dh_onnx PRIVATE ${OpenCV_LIBS})
endif()

target_link_libraries(ultralight_dh_onnx PRIVATE onnxruntime)
