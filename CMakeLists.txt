cmake_minimum_required(VERSION 3.27)
project(NeuralNetworkTest)

set(CMAKE_CXX_STANDARD 17)

# Include CPM for package management
include(get_cpm.cmake)

# Add SFML as a dependency using CPM
CPMAddPackage("gh:SFML/SFML#2.6.1")

# Add the executable with the main file and any additional sources/headers
add_executable(NeuralNetworkTest neural_network.cpp neural_network.h window.h
        main.cpp
        window.cpp)

# Copy necessary data files to the build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Data/set1-images.idx3-ubyte DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Data/set1-labels.idx1-ubyte DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Font/Font.ttf DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
# Define the path to the SFML library directory
set(SFML_DLL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug/_deps/sfml-build/lib")

# Copy all required SFML DLLs to the build directory
file(GLOB SFML_DLLS "${SFML_DLL_PATH}/*.dll")
file(COPY ${SFML_DLLS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Link SFML libraries to the executable
target_link_libraries(NeuralNetworkTest sfml-graphics sfml-window sfml-system)
