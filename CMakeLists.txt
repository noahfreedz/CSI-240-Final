cmake_minimum_required(VERSION 3.27)
project(NeuralNetworkTest)

set(CMAKE_CXX_STANDARD 17)

# Set SFML directory paths
set(SFML_DIR "C:/SFML/lib/cmake/SFML")
include_directories("C:/SFML/include")

# Find SFML package
find_package(SFML 2.5 COMPONENTS graphics window system REQUIRED)

# Add executable
add_executable(NeuralNetworkTest neural_network.cpp neural_network.h)

# Link SFML libraries
target_link_libraries(NeuralNetworkTest sfml-graphics sfml-window sfml-system)
