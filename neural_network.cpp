#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <fstream>
#include <numeric> // For std::accumulate
#include <mutex> // For synchronizing data access
#include <SFML/Graphics.hpp>

#include "neural_network.h"
#include "window.h"

using namespace std;
std::mutex data_mutex;

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

// Function for training a network in a separate thread
void trainNetwork(NeuralNetwork& network, vector<double>& total_errors,
                  vector<unordered_map<int, double>>& weights, vector<unordered_map<int, double>>& biases,
                  const vector<double>& input, const vector<double>& correct_output) {
    total_errors.push_back(network.run_network(input, correct_output));

    // Lock data access to prevent race conditions during weight updates
    std::lock_guard<std::mutex> guard(data_mutex);
    auto network_output = network.backpropigate_network();
    weights.emplace_back(network_output.first);
    biases.emplace_back(network_output.second);
}

int main() {
    // Create Visualizer
    GraphWindow window(1000, 600, "REBECCA");
    window.render();

    // Paths to your MNIST files
    std::string imageFilePath = "train-images.idx3-ubyte";
    std::string labelFilePath = "train-labels.idx1-ubyte";

    // Read images and labels
    int numImages = 200000; // Use a smaller number for quick testing
    int numRows = 28;
    int numCols = 28;

    std::vector<std::vector<double>> images = readMNISTImages(imageFilePath, numImages, numRows, numCols);
    std::vector<int> labels = readMNISTLabels(labelFilePath, numImages);

    // Create networks with different learning rates
    NeuralNetwork networkA(784, 2, 16, 10, 0.05); // Learning rate: 0.05
    NeuralNetwork networkB(784, 2, 16, 10, 0.01); // Learning rate: 0.01
    NeuralNetwork networkC(784, 2, 16, 10, 0.1);  // Learning rate: 0.1

    // Add networks to the visualizer
    window.setLearningRate(0, 0.05);
    window.setLearningRate(1, 0.01);
    window.setLearningRate(2, 0.1);

    vector<unordered_map<int, double>> weights_A, weights_B, weights_C;
    vector<unordered_map<int, double>> biases_A, biases_B, biases_C;

    int count = 0;
    vector<double> total_errors_A, total_errors_B, total_errors_C;

    for (int i = 0; i < images.size(); ++i) {
        vector<double> correct_label_output(10, 0.0);
        correct_label_output[labels[i]] = 1.0;

        // Launch threads for each network
        std::thread threadA(trainNetwork, std::ref(networkA), std::ref(total_errors_A), std::ref(weights_A),
                            std::ref(biases_A), images[i], correct_label_output);

        std::thread threadB(trainNetwork, std::ref(networkB), std::ref(total_errors_B), std::ref(weights_B),
                            std::ref(biases_B), images[i], correct_label_output);

        std::thread threadC(trainNetwork, std::ref(networkC), std::ref(total_errors_C), std::ref(weights_C),
                            std::ref(biases_C), images[i], correct_label_output);

        // Join threads
        threadA.join();
        threadB.join();
        threadC.join();

        count++;

        // Handle Window Events
        window.handleEvents();

        // Print progress every 100 iterations
        if (count % 100 == 0) {
            cout << "RUN (" << count << "/" << "500) - " << endl;
        }

        // Every 500 iterations, average weights, update networks, and visualize cost
        if (count == 500) {
            // Update and visualize network A
            networkA.edit_weights(average(weights_A));
            networkA.edit_biases(average(biases_A));
            double cost_A = networkA.get_cost();
            cout << "GENERATION COMPLETE (Network A) - " << cost_A << endl;
            window.addDataPoint(0, cost_A);

            // Update and visualize network B
            networkB.edit_weights(average(weights_B));
            networkB.edit_biases(average(biases_B));
            double cost_B = networkB.get_cost();
            cout << "GENERATION COMPLETE (Network B) - " << cost_B << endl;
            window.addDataPoint(1, cost_B);

            // Update and visualize network C
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

            count = 0;
        }
        window.render();

    }
    cout << "ENDING PROGRAM" << endl;
    return 0;
}