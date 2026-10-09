cmake_minimum_required(VERSION 3.20)
project(neural_network LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

option(NNET_BUILD_EXAMPLES "Build the reference, Tensor MLP and MNIST examples" ON)
option(NNET_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" OFF)
option(NNET_ENABLE_GLIBCXX_DEBUG "Enable checked libstdc++ containers (changes ABI)" OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE BOOL "Export compiler commands for editors")

if(NNET_ENABLE_SANITIZERS AND NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "The sanitizer option requires GCC or Clang")
endif()

if(NNET_ENABLE_GLIBCXX_DEBUG)
    include(CheckCXXSourceCompiles)
    check_cxx_source_compiles("#include <vector>
        #ifndef __GLIBCXX__
        #error Checked mode needs libstdc++
        #endif
        int main() { return 0; }" NNET_USES_LIBSTDCXX)
    if(NOT NNET_USES_LIBSTDCXX)
        message(FATAL_ERROR "NNET_ENABLE_GLIBCXX_DEBUG requires libstdc++")
    endif()
endif()

# Headers and C++17 are usage requirements. Warnings stay local; the
# checked-container ABI and sanitizer runtime requirements reach consumers.
function(nnet_configure_target target)
    target_include_directories(${target} PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_compile_features(${target} PUBLIC cxx_std_17)
    set_target_properties(${target} PROPERTIES CXX_EXTENSIONS OFF)
    if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
        target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    elseif(MSVC)
        target_compile_options(${target} PRIVATE /W4)
    endif()
    if(NNET_ENABLE_GLIBCXX_DEBUG)
        target_compile_definitions(${target} PUBLIC _GLIBCXX_DEBUG _GLIBCXX_ASSERTIONS)
    endif()
    if(NNET_ENABLE_SANITIZERS)
        target_compile_options(${target} PRIVATE
            -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=undefined)
        target_link_options(${target} PUBLIC -fsanitize=address,undefined)
    endif()
endfunction()

add_library(nnet_core STATIC
    src/core/tensor.cpp
    src/core/tensor_ops.cpp
)
nnet_configure_target(nnet_core)
add_library(nnet::core ALIAS nnet_core)
set_target_properties(nnet_core PROPERTIES EXPORT_NAME core)

add_library(nnet_framework STATIC
    src/autograd/operations.cpp
    src/autograd/backward.cpp
    src/optim/optimizer.cpp
    src/train/trainer.cpp
)
nnet_configure_target(nnet_framework)
target_link_libraries(nnet_framework PUBLIC nnet_core)
add_library(nnet::framework ALIAS nnet_framework)
set_target_properties(nnet_framework PROPERTIES EXPORT_NAME framework)

# The reference can use CSVs without compiling the Tensor adapter.
add_library(nnet_csv STATIC src/dataframe.cpp)
nnet_configure_target(nnet_csv)
add_library(nnet::csv ALIAS nnet_csv)
set_target_properties(nnet_csv PROPERTIES EXPORT_NAME csv)

add_library(nnet_data STATIC
    src/core/tensor_dataframe.cpp
    src/data/mnist.cpp
)
nnet_configure_target(nnet_data)
target_link_libraries(nnet_data PUBLIC nnet_core nnet_csv)
add_library(nnet::data ALIAS nnet_data)
set_target_properties(nnet_data PROPERTIES EXPORT_NAME data)

if(BUILD_TESTING OR NNET_BUILD_EXAMPLES)
    add_library(nnet_reference STATIC src/network.cpp src/training.cpp)
    nnet_configure_target(nnet_reference)
endif()

if(NNET_BUILD_EXAMPLES)
    add_executable(reference_mlp examples/reference_mlp.cpp)
    nnet_configure_target(reference_mlp)
    target_link_libraries(reference_mlp PRIVATE nnet_reference nnet_csv)

    add_executable(tensor_mlp examples/tensor_mlp.cpp)
    nnet_configure_target(tensor_mlp)
    target_link_libraries(tensor_mlp PRIVATE nnet_framework nnet_data)

    add_executable(mnist examples/mnist.cpp)
    nnet_configure_target(mnist)
    target_link_libraries(mnist PRIVATE nnet_framework nnet_data)
endif()

if(BUILD_TESTING)
    add_executable(reference_tests
        tests/reference_test_main.cpp tests/network_test.cpp tests/training_test.cpp)
    target_link_libraries(reference_tests PRIVATE nnet_reference)

    add_executable(tensor_tests
        tests/tensor_test_main.cpp tests/tensor_test.cpp tests/dataframe_test.cpp
        tests/tensor_ops_tests.cpp tests/dense_test.cpp tests/autograd_test.cpp
        tests/trainer_test.cpp tests/mnist_test.cpp)
    target_link_libraries(tensor_tests PRIVATE nnet_framework nnet_data)

    add_executable(parity_tests tests/parity_test_main.cpp tests/parity_test.cpp)
    target_link_libraries(parity_tests PRIVATE nnet_reference nnet_framework)

    add_executable(optimizer_tests tests/optimizer_test.cpp)
    target_link_libraries(optimizer_tests PRIVATE nnet_framework)

    foreach(suite IN ITEMS reference tensor parity optimizer)
        nnet_configure_target(${suite}_tests)
        target_include_directories(${suite}_tests PRIVATE tests)
        set(test_dir "${CMAKE_CURRENT_BINARY_DIR}/test-work/${suite}")
        file(MAKE_DIRECTORY "${test_dir}/tmp")
        add_test(NAME ${suite} COMMAND ${suite}_tests)
        set_tests_properties(${suite} PROPERTIES
            WORKING_DIRECTORY "${test_dir}"
            ENVIRONMENT "TMPDIR=${test_dir}/tmp;TMP=${test_dir}/tmp;TEMP=${test_dir}/tmp"
        )
    endforeach()
    file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/test-work/tensor/data")
    configure_file(data/df_test.csv test-work/tensor/data/df_test.csv COPYONLY)
endif()

# Install reusable code, without the frozen oracle, examples or tests.
install(TARGETS nnet_core nnet_framework nnet_csv nnet_data
    EXPORT nnetTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/nnet DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    FILES_MATCHING PATTERN "*.h")
install(FILES include/dataframe.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

set(nnet_package_dir "${CMAKE_INSTALL_LIBDIR}/cmake/nnet")
configure_package_config_file(cmake/nnetConfig.cmake.in
    "${CMAKE_CURRENT_BINARY_DIR}/nnetConfig.cmake"
    INSTALL_DESTINATION "${nnet_package_dir}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/nnetConfig.cmake"
    DESTINATION "${nnet_package_dir}")
install(EXPORT nnetTargets NAMESPACE nnet:: DESTINATION "${nnet_package_dir}")
