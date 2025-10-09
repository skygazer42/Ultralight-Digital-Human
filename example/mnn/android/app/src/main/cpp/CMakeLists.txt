cmake_minimum_required(VERSION 3.22)
project(dh_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(dh_native SHARED
    digital_human_jni.cpp
    digital_human_engine.cpp
    npy_reader.cpp
)

set(MNN_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../../third_party/MNN CACHE PATH "Path to MNN headers")
if(EXISTS "${MNN_ROOT}/include")
    target_include_directories(dh_native PRIVATE ${MNN_ROOT}/include)
else()
    message(FATAL_ERROR "Please set MNN_ROOT to the directory containing MNN headers")
endif()

set(MNN_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../jniLibs/${ANDROID_ABI})
add_library(MNN SHARED IMPORTED)
set_target_properties(MNN PROPERTIES IMPORTED_LOCATION ${MNN_LIB_DIR}/libMNN.so)

find_package(OpenCV REQUIRED)

target_include_directories(dh_native
    PRIVATE
        ${OpenCV_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(dh_native
    PRIVATE
        MNN
        ${OpenCV_LIBS}
        jnigraphics
        log
        android
)
