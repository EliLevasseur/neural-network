cmake_minimum_required(VERSION 3.20)
project(nnet_consumer LANGUAGES CXX)

find_package(nnet CONFIG REQUIRED)

add_executable(consumer main.cpp)
target_link_libraries(consumer PRIVATE nnet::framework)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(consumer PRIVATE -Wall -Wextra -Wpedantic)
endif()

enable_testing()
add_test(NAME installed_framework COMMAND consumer)
