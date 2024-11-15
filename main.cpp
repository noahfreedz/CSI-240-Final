
#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <fstream>
#include <numeric> // For  accumulate
#include <mutex> // For synchronizing data access

#include "neural_network.h"
#include "window.h"

using namespace std;
using  namespace Rebecca;

vector< vector<double>> readMNISTImages(const  string& filePath, int numImages, int numRows, int numCols) {
     ifstream file(filePath,  ios::binary);
     vector< vector<double>> images;

    if (file.is_open()) {
        int magicNumber = 0;
        int numberOfImages = 0;
        int rows = 0;
        int cols = 0;

        // Read and convert the magic number and header values
        file.read(reinterpret_cast<char*>(&magicNumber), 4);
        file.read(reinterpret_cast<char*>(&numberOfImages), 4);
        file.read(reinterpret_cast<char*>(&rows), 4);
        file.read(reinterpret_cast<char*>(&cols), 4);

        // Convert from big-endian to little-endian if needed
        magicNumber = __builtin_bswap32(magicNumber);
        numberOfImages = __builtin_bswap32(numberOfImages);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);

        for (int i = 0; i < numImages; ++i) {
             vector<double> image;
            for (int j = 0; j < numRows * numCols; ++j) {
                unsigned char pixel = 0;
                file.read(reinterpret_cast<char*>(&pixel), 1);
                image.push_back(static_cast<double>(pixel) / 255.0); // Normalize to [0, 1]
            }
            images.push_back(image);
        }
        file.close();
    } else {
         cerr << "Failed to open the file: " << filePath << "\n";
    }

    return images;
}

vector<int> readMNISTLabels(const  string& filePath, int numLabels) {
     ifstream file(filePath,  ios::binary);
     vector<int> labels;

    if (file.is_open()) {
        int magicNumber = 0;
        int numberOfLabels = 0;

        // Read and convert the magic number and header values
        file.read(reinterpret_cast<char*>(&magicNumber), 4);
        file.read(reinterpret_cast<char*>(&numberOfLabels), 4);

        // Convert from big-endian to little-endian if needed
        magicNumber = __builtin_bswap32(magicNumber);
        numberOfLabels = __builtin_bswap32(numberOfLabels);

        for (int i = 0; i < numLabels; ++i) {
            unsigned char label = 0;
            file.read(reinterpret_cast<char*>(&label), 1);
            labels.push_back(static_cast<int>(label));
        }
        file.close();
    } else {
         cerr << "Failed to open the file: " << filePath << "\n";
    }

    return labels;
}

vector<double> generateStartingWeights(int input_layer, int number_hidden_layers, int number_node_per_hidden, int output_layer) {
     vector<double> startingWeights;

    // Weights for connections from input layer to first hidden layer
    for (int i = 0; i < input_layer * number_node_per_hidden; i++) {
        startingWeights.push_back(getRandom(-1, 1));
    }

    // Weights for connections between hidden layers
    for (int i = 0; i < (number_hidden_layers - 1) * number_node_per_hidden * number_node_per_hidden; i++) {
        startingWeights.push_back(getRandom(-1, 1));
    }

    // Weights for connections from last hidden layer to output layer
    for (int i = 0; i < number_node_per_hidden * output_layer; i++) {
        startingWeights.push_back(getRandom(-1, 1));
    }

    return startingWeights;
}

vector<double> generateStartingBiases(int number_hidden_layers, int number_node_per_hidden, int output_layer) {
     vector<double> startingBiases;

    // Biases for hidden layers
    for (int i = 0; i < number_hidden_layers * number_node_per_hidden; i++) {
        startingBiases.push_back(getRandom(-15.0,15.0));
    }

    // Biases for output layer
    for (int i = 0; i < output_layer; i++) {
        startingBiases.push_back(getRandom(-15.0,15.0));
    }

    return startingBiases;
}

int main() {

    // Paths to MNIST files
     string imageFilePath = "set1-images.idx3-ubyte";
     string labelFilePath = "set1-labels.idx1-ubyte";

    // Read images and labels
    int numImages = 200000;
    int numRows = 28;
    int numCols = 28;

    // setting up vectors for images and labels
    vector< vector<double>> images = readMNISTImages(imageFilePath, numImages, numRows, numCols);
    vector<int> labels = readMNISTLabels(labelFilePath, numImages);

    GraphWindow window(1000, 600, "REBECCA");

    int input_layer = 784;
    int output_layer = 10;
    int number_hidden_layers = 5;
    int number_node_per_hidden = 32;

    vector<double> startingWeights = generateStartingWeights(input_layer, number_hidden_layers, number_node_per_hidden, output_layer);
    vector<double> startingBiases = generateStartingBiases(number_hidden_layers, number_node_per_hidden, output_layer);

    int count = 0;
    ThreadNetworks allNetworks(window, 5, 0.01, 10, startingWeights, startingBiases, input_layer,
              number_hidden_layers,number_node_per_hidden, output_layer);


    while (true) {
        int i = getRandom(0, images.size());

        vector<double> correct_label_output(10, 0.0);
        correct_label_output[labels[i]] = 1.0;

        allNetworks.runThreading(images[i], correct_label_output);

        count++;

        if (count == 100) {
            allNetworks.PrintCost();
            count = 0;
        }

        // Render
        window.render();
        window.handleEvents();
    }

    return 0;
}