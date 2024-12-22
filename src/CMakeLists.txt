# Collect all source files in this directory
file(GLOB SOURCES "*.cpp")

add_definitions(-DURL="${URL}")

# Add source file
add_executable(${PROJECT_NAME} ${SOURCES})

# Link required libraries (e.g., curl)
find_package(CURL REQUIRED)
target_link_libraries(${PROJECT_NAME} CURL::libcurl)