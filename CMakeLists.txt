cmake_minimum_required(VERSION 3.23.1)

project(
  cudf-demo
  VERSION 0.0.1
  LANGUAGES CXX CUDA
)

find_package(cudf REQUIRED)

# add_executable(cudf-demo 
#                src/main.cc
#                src/to_arrow_device_array.cc)
add_library(cudf-demo SHARED
            src/main.cc            
            src/to_arrow_device_array.cc)
target_link_libraries(cudf-demo PRIVATE cudf::cudf)
target_compile_features(cudf-demo PRIVATE cxx_std_20)
