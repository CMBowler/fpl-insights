# Minimum CMake version required
cmake_minimum_required(VERSION 3.10)

# Project name and language
get_filename_component(PROJECT_NAME ${CMAKE_CURRENT_SOURCE_DIR} NAME)
project(${PROJECT_NAME} LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Include directories for headers
include_directories(${CMAKE_SOURCE_DIR}/inc)

# Add subdirectory for source files
add_subdirectory(src)
