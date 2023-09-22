cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_CXX_STANDARD 17)
project(xs_inference_examples VERSION 1.0.0 LANGUAGES C CXX)

set(PROJECT_ROOT ${PROJECT_SOURCE_DIR})
set(PROJECT_INCLUDE "${PROJECT_ROOT}/include")
set(PROJECT_SRC "${PROJECT_ROOT}/src")
set(PROJECT_MODELS "${PROJECT_SRC}/models")
set(PROJECT_THIRD_PARTY "${PROJECT_ROOT}/third_party")

execute_process(COMMAND protoc -I ${PROJECT_THIRD_PARTY}/onnx/onnx --cpp_out=${PROJECT_INCLUDE}/xs_inference_examples onnx.proto3)

IF(NOT DEFINED GOOGLEBENCHMARK_SOURCE_DIR)
    MESSAGE(STATUS "Downloading Google Benchmark to ${CMAKE_BINARY_DIR}/googlebenchmark-source (define GOOGLEBENCHMARK_SOURCE_DIR to avoid it)")
    CONFIGURE_FILE(cmake/DownloadGoogleBenchmark.cmake "${CMAKE_BINARY_DIR}/googlebenchmark-download/CMakeLists.txt")
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" -G "${CMAKE_GENERATOR}" .
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googlebenchmark-download" )
    EXECUTE_PROCESS(COMMAND "${CMAKE_COMMAND}" --build .
            WORKING_DIRECTORY "${CMAKE_BINARY_DIR}/googlebenchmark-download")
    SET(GOOGLEBENCHMARK_SOURCE_DIR "${CMAKE_BINARY_DIR}/googlebenchmark-source" CACHE STRING "Google Benchmark source directory")
ENDIF()

add_subdirectory(
        ${PROJECT_THIRD_PARTY}/spdlog
)

include_directories(
        ${PROJECT_INCLUDE}
        ${PROJECT_THIRD_PARTY}/spdlog/include
        ${PROJECT_ROOT}/weights
        ${GOOGLEBENCHMARK_SOURCE_DIR}/include
)

set(onnx_src_common ${PROJECT_INCLUDE}/xs_inference_examples/onnx.proto3.pb.cc)



# Set flags to build type
set(EXTRA_C_FLAGS "${EXTRA_C_FLAGS} -Wall -Wpedantic -Wno-narrowing")
set(EXTRA_C_FLAGS_DEBUG   "${EXTRA_C_FLAGS_DEBUG} -g3 -pg -fsanitize=address -fsanitize=leak")

# Set compiler options
set(CMAKE_CXX_FLAGS         "${CMAKE_CXX_FLAGS} ${EXTRA_C_FLAGS}")
set(CMAKE_CXX_FLAGS_DEBUG   "${CMAKE_CXX_FLAGS_DEBUG} ${EXTRA_C_FLAGS_DEBUG} -O0 -fopenmp")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} ${EXTRA_C_FLAGS_DEBUG} -ffast-math")



add_executable(codegen ${PROJECT_SRC}/codegen.cc ${onnx_src_common})
target_link_libraries(codegen protobuf absl_log_internal_message absl_log_internal_check_op xsdnn)

add_library(xs_models
        ${PROJECT_MODELS}/ssd_mobilenet_v1_1_default_1.cc)

add_executable(label_image ${PROJECT_SRC}/label_image.cc)
target_link_libraries(label_image xsdnn xs_models protobuf absl_log_internal_message absl_log_internal_check_op)

IF(NOT TARGET benchmark)
    SET(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "")
    ADD_SUBDIRECTORY(
            "${GOOGLEBENCHMARK_SOURCE_DIR}"
            "${CMAKE_BINARY_DIR}/deps/googlebenchmark")
ENDIF()

set(BENCH_UTILS ${PROJECT_SRC}/bench/utils.cc)

add_executable(f32_conv_bench ${PROJECT_SRC}/bench/f32_conv.cc ${BENCH_UTILS})
target_link_libraries(f32_conv_bench benchmark xsdnn protobuf)