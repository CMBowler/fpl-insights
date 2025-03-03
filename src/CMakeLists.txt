# Collect all source files in this directory
file(GLOB SOURCES "*.cpp")

add_definitions(-DURL="${URL}")

# Add source file
add_executable(${PROJECT_NAME} ${SOURCES})

# Link required libraries
find_package(CURL REQUIRED)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

# Include Python headers
target_include_directories(${PROJECT_NAME} PRIVATE ${Python3_INCLUDE_DIRS})

# Link Libraries
target_link_libraries(${PROJECT_NAME} CURL::libcurl ${Python3_LIBRARIES})