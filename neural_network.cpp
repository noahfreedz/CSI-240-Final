#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <fstream>
#include <numeric> // For  accumulate
#include <mutex> // For synchronizing data access
#include <SFML/Graphics.hpp>

#include "neural_network.h"
#include "window.h"

using namespace std;
using  namespace Rebecca;

 mutex data_mutex;

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

unordered_map<int, double> average(vector<unordered_map<int, double>>& _vector)
{
    unordered_map<int, double> averagedWeights;
    int count = 0;

    for (const auto& base : _vector) {
        count++;
        for (const auto& pair : base) {
            averagedWeights[pair.first] += pair.second;
        }
    }

    for (auto& weight : averagedWeights) {
        weight.second = weight.second / count;
    }

    return averagedWeights;
}

double averageError(const std::vector<double>& values) {
     if (values.empty()) {
         return 0.0; // Return 0 if the vector is empty to avoid division by zero
     }

     double sum = std::accumulate(values.begin(), values.end(), 0.0);
     return sum / values.size();
 }

void trainNetwork(NeuralNetwork& network, vector<double>& total_errors, vector<unordered_map<int, double>>& _weights,vector<unordered_map<int, double>>& _bias, const vector<double>& input, const vector<double>& correct_output) {


    total_errors.push_back(network.run_network(input, correct_output));
    // Lock data access to prevent race conditions during weight updates
     lock_guard< mutex> guard(data_mutex);
     auto network_return = network.backpropigate_network();
     _weights.push_back(network_return.first);
     _bias.push_back(network_return.second);
}

int main() {
    // Create Visualizer

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
    window.render();

    int input_layer = 784;
    int output_layer = 10;
    int number_hidden_layers = 2;
    int number_node_per_hidden = 16;

     vector<double> startingWeights = generateStartingWeights(input_layer, number_hidden_layers, number_node_per_hidden, output_layer);
     vector<double> startingBiases = generateStartingBiases(number_hidden_layers, number_node_per_hidden, output_layer);

    // Create networks with different learning rates
    NeuralNetwork networkA(input_layer, number_hidden_layers, number_node_per_hidden, output_layer, 0.05, startingWeights,startingBiases); // Learning rate: 0.05
    NeuralNetwork networkB(input_layer, number_hidden_layers, number_node_per_hidden, output_layer, 0.01, startingWeights,startingBiases); // Learning rate: 0.01
    NeuralNetwork networkC(input_layer, number_hidden_layers, number_node_per_hidden, output_layer, 0.1, startingWeights,startingBiases);  // Learning rate: 0.1

    // Add networks to the visualizer
    window.setLearningRate(0, 0.05);
    window.setLearningRate(1, 0.01);
    window.setLearningRate(2, 0.1);

     int count = 0;

    vector<unordered_map<int, double>> weights_A, weights_B, weights_C;
    vector<unordered_map<int, double>> biases_A, biases_B, biases_C;

    vector<double> total_errors_A, total_errors_B, total_errors_C;
    while (true) {
        int i = getRandom(0, images.size());

        vector<double> correct_label_output(10, 0.0);
        correct_label_output[labels[i]] = 1.0;

        // Launch threads for each network
         thread threadA(trainNetwork,  ref(networkA),  ref(total_errors_A),  ref(weights_A),
                             ref(biases_A), images[i], correct_label_output);

         thread threadB(trainNetwork,  ref(networkB),  ref(total_errors_B),  ref(weights_B),
                             ref(biases_B), images[i], correct_label_output);

         thread threadC(trainNetwork,  ref(networkC),  ref(total_errors_C),  ref(weights_C),
                             ref(biases_C), images[i], correct_label_output);

        // Join threads
        threadA.join();
        threadB.join();
        threadC.join();

        count++;

        // Handle Window Events
        window.handleEvents();

        // Every 500 iterations, average weights, update networks, and visualize cost
        if (count == 100) {
            // Update and visualize network A
            networkA.edit_weights(average(weights_A));
            networkA.edit_biases(average(biases_A));
            double cost_A = networkA.getCost();
            cout << "GENERATION COMPLETE (Network A) - " << cost_A << endl;
            window.addDataPoint(0, cost_A);

            // Update and visualize network B
            networkB.edit_weights(average(weights_B));
            networkB.edit_biases(average(biases_B));
            double cost_B = networkB.getCost();
            cout << "GENERATION COMPLETE (Network B) - " << cost_B << endl;
            window.addDataPoint(1, cost_B);

            // Update and visualize network C
            networkC.edit_weights(average(weights_C));
            networkC.edit_biases(average(biases_C));
            double cost_C = networkC.getCost();
            cout << "GENERATION COMPLETE (Network C) - " << cost_C << endl;
            window.addDataPoint(2, cost_C);

            // Clear data and render window
            weights_A.clear();
            weights_B.clear();
            weights_C.clear();
            biases_A.clear();
            biases_B.clear();
            biases_C.clear();
            total_errors_A.clear();
            total_errors_B.clear();
            total_errors_C.clear();

            count = 0;
        }

        window.render();

    }

    return 0;
}