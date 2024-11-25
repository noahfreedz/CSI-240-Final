cmake_minimum_required(VERSION 3.27)

# Set the project name
project(NeuralNetworkProjects)

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)

# Include CPM for package management
include(get_cpm.cmake)

# Add SFML as a dependency using CPM
CPMAddPackage("gh:SFML/SFML#2.6.1")

# Define the path to the SFML library directory
set(SFML_DLL_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake-build-debug/_deps/sfml-build/lib")

# Copy all required SFML DLLs to the build directory
file(GLOB SFML_DLLS "${SFML_DLL_PATH}/*.dll")
file(COPY ${SFML_DLLS} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Copy necessary data files to the build directory
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Data/set1-images.idx3-ubyte DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Data/set1-labels.idx1-ubyte DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Data/Testing-Data-Images.idx3-ubyte DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Data/TestingData-labels.idx1-ubyte DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/Font/Font.ttf DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

# Add the MakeNewNetwork executable
add_executable(MakeNewNetwork
        Make_New_Netwroks.cpp
        neural_network.cpp
        window.cpp
)
target_link_libraries(MakeNewNetwork sfml-graphics sfml-window sfml-system)

# Add the FineTune executable
add_executable(FineTune
        fineTune.cpp
        neural_network.cpp
        window.cpp
)
target_link_libraries(FineTune sfml-graphics sfml-window sfml-system)

# Add the GetBest executable
add_executable(GetBest
        FileSorting.cpp
        neural_network.cpp
        window.cpp
)
target_link_libraries(GetBest sfml-graphics sfml-window sfml-system)

# Specify common include directories
target_include_directories(MakeNewNetwork PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(FineTune PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_include_directories(GetBest PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
