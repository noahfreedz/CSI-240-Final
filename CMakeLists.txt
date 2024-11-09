cmake_minimum_required(VERSION 3.27)
project(NeuralNetworkTest)

set(CMAKE_CXX_STANDARD 17)
include(get_cpm.cmake)

CPMAddPackage("gh:SFML/SFML#2.6.1")

# Add executable
add_executable(NeuralNetworkTest neural_network.cpp neural_network.h window.h
        window.h)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/train-images.idx3-ubyte DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/train-labels.idx1-ubyte DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


# Link SFML libraries
target_link_libraries(NeuralNetworkTest sfml-graphics sfml-window sfml-system)
