@echo off

REM Make sure you have CMake and a compatible compiler 
REM installed and properly configured in your system's PATH

REM Remove the build directory if it exists
rmdir /s /q build

REM Create a new build directory
mkdir build
cd build

REM Run CMake to configure the project
cmake ..

REM Build the project
cmake --build .

REM Run the program with the specified arguments
fpl-predict.exe "fantasy.premierleague.com/api/"
