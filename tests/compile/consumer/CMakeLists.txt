cmake_minimum_required(VERSION 3.24)

project(numabraid_consumer LANGUAGES CXX)

find_package(numabraid CONFIG REQUIRED)

add_executable(numabraid_consumer main.cc)
target_link_libraries(numabraid_consumer PRIVATE numabraid::numabraid)
