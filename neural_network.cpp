#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <fstream>
#include <numeric> // For std::accumulate
#include <SFML/Graphics.hpp>

#include "neural_network.h"
#include "window.h"

using namespace std;

std::vector<std::vector<double>> readMNISTImages(const std::string& filePath, int numImages, int numRows, int numCols) {
    std::ifstream file(filePath, std::ios::binary);
    std::vector<std::vector<double>> images;

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
            std::vector<double> image;
            for (int j = 0; j < numRows * numCols; ++j) {
                unsigned char pixel = 0;
                file.read(reinterpret_cast<char*>(&pixel), 1);
                image.push_back(static_cast<double>(pixel) / 255.0); // Normalize to [0, 1]
            }
            images.push_back(image);
        }
        file.close();
    } else {
        std::cerr << "Failed to open the file: " << filePath << "\n";
    }

    return images;
}

std::vector<int> readMNISTLabels(const std::string& filePath, int numLabels) {
    std::ifstream file(filePath, std::ios::binary);
    std::vector<int> labels;

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
        std::cerr << "Failed to open the file: " << filePath << "\n";
    }

    return labels;
}

unordered_map<int, double> average(vector<unordered_map<int, double>>& _vector) {
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

int main() {
    // Create Visualizer
    GraphWindow window(1000, 600, "REBECCA");
    window.render();

    // Paths to your MNIST files
    std::string imageFilePath = "train-images.idx3-ubyte";
    std::string labelFilePath = "train-labels.idx1-ubyte";

    // Read images and labels
    int numImages = 100000; // Change this to read as many as you need
    int numRows = 28;
    int numCols = 28;

    std::vector<std::vector<double>> images = readMNISTImages(imageFilePath, numImages, numRows, numCols);
    std::vector<int> labels = readMNISTLabels(labelFilePath, numImages);

    // Create and train the networks
    NeuralNetwork networkA(784, 2, 16, 10, 0.05); // Learning rate: 0.05
    NeuralNetwork networkB(784, 2, 16, 10, 0.01); // Learning rate: 0.01
    NeuralNetwork networkC(784, 2, 16, 10, 0.1);  // Learning rate: 0.1

    // Add networks to the visualizer with learning rates
    window.setLearningRate(0, 0.05);
    window.setLearningRate(1, 0.01);
    window.setLearningRate(2, 0.1);

    vector<unordered_map<int, double>> weights_A, weights_B, weights_C;
    vector<unordered_map<int, double>> biases_A, biases_B, biases_C;

    int count = 0;
    vector<double> total_errors_A, total_errors_B, total_errors_C;

    for (int i = 0; i < images.size(); ++i) {
        vector<double> correct_label_output(10, 0.0);
        correct_label_output[labels[i]] = 1.0; // Set the correct output to 1.0 for the label

        // Run networks and collect errors
        total_errors_A.push_back(networkA.run_network(images[i], correct_label_output));
        total_errors_B.push_back(networkB.run_network(images[i], correct_label_output));
        total_errors_C.push_back(networkC.run_network(images[i], correct_label_output));

        // Perform backpropagation and store weights/biases for averaging
        weights_A.emplace_back(networkA.backpropigate_network().first);
        biases_A.emplace_back(networkA.backpropigate_network().second);

        weights_B.emplace_back(networkB.backpropigate_network().first);
        biases_B.emplace_back(networkB.backpropigate_network().second);

        weights_C.emplace_back(networkC.backpropigate_network().first);
        biases_C.emplace_back(networkC.backpropigate_network().second);

        count++;

        // Handle Window Events
        window.handleEvents();

        // Print progress every 100 iterations
        if (count % 100 == 0) {
            cout << "RUN (" << count << "/" << "500) - " << endl;
        }

        // Every 500 iterations, average weights, update networks, and visualize cost
        if (count == 500) {
            // Update network A
            networkA.edit_weights(average(weights_A));
            networkA.edit_biases(average(biases_A));
            double cost_A = networkA.get_cost();
            cout << "GENERATION COMPLETE (Network A) - " << cost_A << endl;
            window.addDataPoint(0, cost_A);

            // Update network B
            networkB.edit_weights(average(weights_B));
            networkB.edit_biases(average(biases_B));
            double cost_B = networkB.get_cost();
            cout << "GENERATION COMPLETE (Network B) - " << cost_B << endl;
            window.addDataPoint(1, cost_B);

            // Update network C
            networkC.edit_weights(average(weights_C));
            networkC.edit_biases(average(biases_C));
            double cost_C = networkC.get_cost();
            cout << "GENERATION COMPLETE (Network C) - " << cost_C << endl;
            window.addDataPoint(2, cost_C);

            // Clear data and render window
            weights_A.clear();
            biases_A.clear();
            weights_B.clear();
            biases_B.clear();
            weights_C.clear();
            biases_C.clear();
            total_errors_A.clear();
            total_errors_B.clear();
            total_errors_C.clear();

            window.render();
            count = 0;
        }
    }

    return 0;
}
