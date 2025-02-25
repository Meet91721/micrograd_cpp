project(micrograd)
cmake_minimum_required(VERSION 3.16)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# For making sure that compile_commands.json file is generated
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# GoogleTest requires at least C++14
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Includes all the header files
include_directories(${CMAKE_SOURCE_DIR}/include)

# Creates a shared library from Value.cpp
add_library(Value SHARED lib/Value.cpp)

# for every executable file you need to provide a bunch of src files
add_executable(main src/main.cpp)

# linking executable file with the shared library -> Value
target_link_libraries(main PRIVATE Value)

# For support of google-test libraries
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)

set(gtest_force_shared_crt
    ON
    CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Creates a executable file named main_test from the operations.cpp
add_executable(main_test tests/operations.cpp)

# links the gtest_main and Value lib with main_test
target_link_libraries(main_test PRIVATE Value GTest::gtest_main)

# includes the module GoogleTest
include(GoogleTest)
gtest_discover_tests(main_test)
