#include <iostream>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <fstream>
#include <vector>
#include <iostream>
#include "neural_network.h"
#include "window.h"
#include <SFML/Graphics.hpp>

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

    // Create and train the network
    NeuralNetwork networkA(784, 2, 16, 10, 0.05); // 784 input nodes for 28x28 images
    window.setLearningRate(0, 0.05);
    vector<unordered_map<int, double>> weights_A;
    vector<unordered_map<int, double>> biases_A;


    int count = 0;
    vector<double> total_errors_A;

    for (int i = 0; i < images.size(); ++i) {
        vector<double> correct_label_output;
        correct_label_output.resize(10);
        for(int output = 0; output < 10; output++) {
            if(labels[i] == output) {
                correct_label_output[output] = 1.0;
            } else {
                correct_label_output[output] = 0.0;
            }
        }
        total_errors_A.push_back(networkA.run_network(images[i], correct_label_output));

        pair<unordered_map<int, double>, unordered_map<int, double>> network_outputA = networkA.backpropigate_network();
        weights_A.emplace_back(network_outputA.first);
        biases_A.emplace_back(network_outputA.second);

        count++;

        // Handle Window Events
        window.handleEvents();
        // Average weights if necessary
        if(count % 100 == 0) {
            cout << "RUN (" << count << "/" << "500) - " << endl;
        }
        if(count == 500)
        {
            unordered_map<int, double> averaged_weights_A = average(weights_A);
            unordered_map<int, double> averaged_biases_A= average(biases_A);
            networkA.edit_weights(averaged_weights_A);
            networkA.edit_biases(averaged_biases_A);
            weights_A.clear();
            biases_A.clear();
            averaged_weights_A.clear();
            averaged_biases_A.clear();
            cout << "GENERATION COMPLETE - " << networkA.get_cost() << endl;
            window.addDataPoint(0, networkA.get_cost());
            window.render();
            //std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            total_errors_A.clear();
            count = 0;
        }
    }

    return 0;
}