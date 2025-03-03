#!/bin/bash

# Create a build directory
mkdir -p build
cd build

# Run CMake to configure the project
cmake ..

# Build the project
make

cd ..

# Run the program with the url argument
./build/fpl-predict "fantasy.premierleague.com/api/"

