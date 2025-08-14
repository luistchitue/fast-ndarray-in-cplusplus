cmake_minimum_required(VERSION 3.10)
project(fast_nd_array_in_cpp)

find_package(Catch2 CONFIG REQUIRED)

add_executable(fast_nd_array_in_cpp ./tests/unit_test.cpp)
target_link_libraries(fast_nd_array_in_cpp PRIVATE Catch2::Catch2WithMain)