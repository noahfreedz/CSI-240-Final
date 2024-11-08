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


int main() {
    // Paths to your MNIST files
    std::string imageFilePath = "C:\\Users\\nfree\\OneDrive\\Desktop\\SCOTTS-BRANCH\\cmake-build-debug\\train-images.idx3-ubyte";
    std::string labelFilePath = "C:\\Users\\nfree\\OneDrive\\Desktop\\SCOTTS-BRANCH\\cmake-build-debug\\train-labels.idx1-ubyte";

    // Read images and labels
    int numImages = 100000; // Change this to read as many as you need
    int numRows = 28;
    int numCols = 28;

    std::vector<std::vector<double>> images = readMNISTImages(imageFilePath, numImages, numRows, numCols);
    std::vector<int> labels = readMNISTLabels(labelFilePath, numImages);

    // Create and train the network
    NeuralNetwork myNN(784, 2, 16, 10); // 784 input nodes for 28x28 images
    std::vector<std::unordered_map<int, double>> allWeights;
    int count = 0;
    for (int i = 0; i < images.size(); ++i) {
        vector<double> correct_outputs;
        correct_outputs.reserve(10);
        for(int output = 0; output < 10; output++) {
            cout << "OUTPUT: " << output << " - LABEL: " << labels[i] << endl;
            if(labels[i] == output) {
                correct_outputs[output] = 1.0;
            } else {
                correct_outputs[output] = 0.0;
            }
        }
        myNN.run_network(images[i], correct_outputs);
        std::unordered_map<int, double> newWeights = myNN.backpropigate_network(); // Use the label as the correct node
        allWeights.emplace_back(newWeights);
        count++;

        cout << "The Number of tryes is: " << myNN.trys << " The Number Correct is :" << myNN.correct << endl;
        // Average weights if necessary
        if(count == 100)
        {
            std::unordered_map<int, double> averagedWeights = average(allWeights);
            myNN.edit_weights(averagedWeights);
            allWeights.clear();
            averagedWeights.clear();
            cout << "Editing Weights!" << endl;
            std::this_thread::sleep_for(std::chrono::seconds(1));
            count = 0;
        }
    }

    return 0;
}