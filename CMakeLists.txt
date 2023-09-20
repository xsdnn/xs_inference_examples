cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_CXX_STANDARD 17)
project(xs_inference_examples VERSION 1.0.0 LANGUAGES C CXX)

set(PROJECT_ROOT ${PROJECT_SOURCE_DIR})
set(PROJECT_INCLUDE "${PROJECT_ROOT}/include")
set(PROJECT_SRC "${PROJECT_ROOT}/src")
set(PROJECT_MODELS "${PROJECT_SRC}/models")
set(PROJECT_THIRD_PARTY "${PROJECT_ROOT}/third_party")

execute_process(COMMAND protoc -I ${PROJECT_THIRD_PARTY}/onnx/onnx --cpp_out=${PROJECT_INCLUDE}/xs_inference_examples onnx.proto3)

add_subdirectory(
        ${PROJECT_THIRD_PARTY}/spdlog
)

include_directories(
        ${PROJECT_INCLUDE}
        ${PROJECT_THIRD_PARTY}/spdlog/include
)

set(onnx_src_common ${PROJECT_INCLUDE}/xs_inference_examples/onnx.proto3.pb.cc)

add_library(xs_models
        ${PROJECT_MODELS}/ssd_mobilenet_v1_1_default_1.cc)

add_executable(label_image ${PROJECT_SRC}/label_image.cc)

add_executable(codegen ${PROJECT_SRC}/codegen.cc ${PROJECT_INCLUDE}/xs_inference_examples/onnx.proto3.pb.cc)
target_link_libraries(codegen protobuf absl_log_internal_message absl_log_internal_check_op)