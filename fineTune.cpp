
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

    // Paths to MNIST files
     string imageFilePath = "set1-images.idx3-ubyte";
     string labelFilePath = "set1-labels.idx1-ubyte";

    // Read images and labels
    int numImages = 60000;
    int numRows = 28;
    int numCols = 28;

    // setting up vectors for images and labels
    vector< vector<double>> images = readMNISTImages(imageFilePath, numImages, numRows, numCols);
    vector<int> labels = readMNISTLabels(labelFilePath, numImages);

    int input_layer = 784;
    int output_layer = 10;
    int number_hidden_layers = 2;
    int number_node_per_hidden = 128;
    int runs_tell_backprop = 100;

    // vector<double> startingWeights = generateStartingWeights(input_layer, number_hidden_layers, number_node_per_hidden, output_layer);
    // vector<double> startingBiases = generateStartingBiases(number_hidden_layers, number_node_per_hidden, output_layer);

    int count = 0;
    ThreadNetworks allNetworks(5, .01, .1,  input_layer,
              number_hidden_layers,number_node_per_hidden,
              output_layer, DIR+"1.820964/Network.bin", runs_tell_backprop);

    GraphWindow window_(1000, 600, "REBECCA", &allNetworks);

    allNetworks.SetWindow(window_);
    int i = 0;
    while (window_.run_network ) {
        i = getRandom(0, numImages);
        vector<double> correct_label_output(10, 0.0);
        correct_label_output[labels[i]] = 1.0;
        allNetworks.runThreading(images[i], correct_label_output);
        count++;

        if (count == runs_tell_backprop) {
            allNetworks.PrintCost();
            count = 0;
        }

        window_.render();
        window_.handleEvents();
    }
    return 0;
}