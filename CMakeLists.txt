# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and language
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME} LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Source files in src directory
set(SOURCES
    main.cpp
    fetch.cpp
    form.cpp
    score.cpp
)

add_definitions(-DURL="${URL}")

# Add source file
add_executable(${PROJECT_NAME} ${SOURCES})

# Link required libraries (e.g., curl)
find_package(CURL REQUIRED)
target_link_libraries(${PROJECT_NAME} CURL::libcurl)

# Include directories for external libraries (JSON for Modern C++)
#include_directories(${CMAKE_SOURCE_DIR}/include)

