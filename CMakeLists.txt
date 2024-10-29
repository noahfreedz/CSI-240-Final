cmake_minimum_required(VERSION 3.27)
project(NeuralNetworkTest)

set(CMAKE_CXX_STANDARD 17)

include_directories(c:/SFML/include)

set(SFML_DIR "C:/SFML/lib/cmake/SFML")
# Find SFML package
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# Add executable before linking
add_executable(NeuralNetworkTest main.cpp main.h
        window.h)

# Link SFML libraries
target_link_libraries(NeuralNetworkTest sfml-graphics sfml-window sfml-system)
