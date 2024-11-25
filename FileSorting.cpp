
#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <fstream>
#include <queue>

#include "neural_network.h"

using namespace std;
using  namespace Rebecca;

string DIR = "../network/";

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

int main() {

    // Paths to MNIST files
     string imageFilePath = "Testing-Data-Images.idx3-ubyte";
     string labelFilePath = "TestingData-labels.idx1-ubyte";

    queue<string> filePaths;

    // Read images and labels
    int numImages = 200000;
    int numRows = 28;
    int numCols = 28;

    // setting up vectors for images and labels
    vector<vector<double>> images = readMNISTImages(imageFilePath, numImages, numRows, numCols);
    vector<int> labels = readMNISTLabels(labelFilePath, numImages);

    int input_layer = 784;
    int output_layer = 10;
    int number_hidden_layers = 2;
    int number_node_per_hidden = 128;

    for (const auto& entry : directory_iterator(DIR)) {
        filePaths.push(entry.path().filename().string());
    }

    unordered_map<string, int> Points;

    while(!filePaths.empty())
    {
        int points = 0;
        string filePath = filePaths.front();
        filePaths.pop();


        NeuralNetwork network(input_layer, number_hidden_layers,number_node_per_hidden,
                  output_layer, 1.0, DIR + filePath +"/Network.bin");


        if(network.getCost() <= 1.0) {
            points += 2;
        }

        for(int i = 0; i < 28; i ++) {
            vector<double> correct_label_output(10, 0.0);
            correct_label_output[labels[i]] = 1.0;
            network.run_network(images[i], correct_label_output);
        }

        points += network.precise_correct_count*10;
        points += network.vauge_correct_count*5;

        Points[filePath] = points;
    };

    for(auto net : Points) {
        cout << "File Name :" << net.first << " Number Points :"<< net.second << "\n";
    }



    return 0;
}