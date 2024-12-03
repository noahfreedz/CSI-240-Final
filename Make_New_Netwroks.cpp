
#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <fstream>

#include "neural_network.h"

using namespace std;
using  namespace Rebecca;

int main() {
    string imageFilePath = "set1-images.idx3-ubyte";
    string labelFilePath = "set1-labels.idx1-ubyte";

    int numImages = 60000;
    int numRows = 28;
    int numCols = 28;

    vector<vector<double>> images = readMNISTImages(imageFilePath, numImages, numRows, numCols);
    vector<int> labels = readMNISTLabels(labelFilePath, numImages);

    int input_layer = 784;  // 28x28 pixels
    int output_layer = 10;  // 10 digits
    int number_hidden_layers = 3;
    int number_node_per_hidden = 512;
    int runs_tell_backprop = 100;

    // Generate weights and biases with improved initialization
    vector<double> startingWeights = generateStartingWeights(input_layer, number_hidden_layers,
                                                           number_node_per_hidden, output_layer);
    vector<double> startingBiases = generateStartingBiases(number_hidden_layers,
                                                         number_node_per_hidden, output_layer);

    // Use smaller learning rates for CCE loss
    ThreadNetworks allNetworks(4, 0.0001, 0.001, startingWeights, startingBiases,
                             input_layer, number_hidden_layers, number_node_per_hidden,
                             output_layer, runs_tell_backprop);

    GraphWindow window_(1000, 600, "REBECCA", &allNetworks);
    allNetworks.SetWindow(window_);

    while (window_.run_network) {
        int i = getRandom(0, numImages - 1);
        vector<double> correct_label_output(10, 0.0);
        correct_label_output[labels[i]] = 1.0;
        allNetworks.runThreading(images[i], correct_label_output);
        window_.render();
        window_.handleEvents();
    }

    return 0;
}