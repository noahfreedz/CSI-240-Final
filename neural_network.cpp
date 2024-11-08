#include <iostream>
#include <cstdlib>
#include <ctime>
#include "neural_network.h"
#include <vector>
#include <unordered_map>
#include <chrono>
#include <thread>

//#include "window.h"
//#include <SFML/Graphics.hpp>

using namespace std;

#include <fstream>
#include <vector>
#include <iostream>

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
    // Paths to your MNIST files
    std::string imageFilePath = "C:\\coding shit\\AI\\CSI-240-Final\\train-images.idx3-ubyte";
    std::string labelFilePath = "C:\\coding shit\\AI\\CSI-240-Final\\train-labels.idx1-ubyte";

    // Read images and labels
    int numImages = 100000; // Change this to read as many as you need
    int numRows = 28;
    int numCols = 28;

    std::vector<std::vector<double>> images = readMNISTImages(imageFilePath, numImages, numRows, numCols);
    std::vector<int> labels = readMNISTLabels(labelFilePath, numImages);

    // Create and train the network
    NeuralNetwork myNN(784, 2, 16, 10); // 784 input nodes for 28x28 images
    vector<unordered_map<int, double>> all_weights;
    vector<unordered_map<int, double>> all_biases;
    int count = 0;
    vector<double> total_errors;
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
        total_errors.push_back(myNN.run_network(images[i], correct_label_output));
        pair<unordered_map<int, double>, unordered_map<int, double>> network_output = myNN.backpropigate_network(); // Use the label as the correct node
        all_weights.emplace_back(network_output.first);
        all_biases.emplace_back(network_output.second);
        count++;

        // Average weights if necessary
        if(count % 100 == 0) {
            cout << "RUN (" << count << "/" << "500) - " << averageError(total_errors) << endl;
        }
        if(count == 100)
        {
            unordered_map<int, double> averaged_weights = average(all_weights);
            unordered_map<int, double> averaged_biases = average(all_biases);
            myNN.edit_weights(averaged_weights);
            myNN.edit_biases(averaged_biases);
            all_weights.clear();
            all_biases.clear();
            averaged_biases.clear();
            averaged_weights.clear();
            cout << "GENERATION COMPLETE - " << myNN.getCost() << endl;
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            total_errors.clear();
            count = 0;
        }
    }

    return 0;
}