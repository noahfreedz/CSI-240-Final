
#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <fstream>

#include "neural_network.h"

using namespace std;
using  namespace Rebecca;

struct Point_System {
    int _precise_correct_count = 0;
    int _vauge_correct_count = 0;
    float _cost = 0.0f;
    int _total_points = 0;

    Point_System() = default;

    Point_System(int precise_correct_count, int vauge_correct_count, float cost)
        : _precise_correct_count(precise_correct_count), _vauge_correct_count(vauge_correct_count), _cost(cost) {
        _total_points += _precise_correct_count * 10;
        _total_points += _vauge_correct_count * 5;
        if (_cost <= 1.0) {
            _total_points += 2;
        }
    }

    bool operator>=(int i) const {
        return _total_points >= i;
    }
};

ostream& operator<<(ostream& lhs, const Point_System& rhs){
    lhs << "Total Points: " << rhs._total_points
        << " Precise: " << rhs._precise_correct_count
        << " Vague: " << rhs._vauge_correct_count
        << " Cost: " << rhs._cost;
    return lhs;
}



int main() {
    // Paths to MNIST files
    string imageFilePath = "Testing-Data-Images.idx3-ubyte";
    string labelFilePath = "TestingData-labels.idx1-ubyte";

    queue<string> filePaths;

    // Read images and labels
    int numImages = 60000;
    int numRows = 28;
    int numCols = 28;

    // setting up vectors for images and labels
    vector<vector<double>> images = readMNISTImages(imageFilePath, numImages, numRows, numCols);
    vector<int> labels = readMNISTLabels(labelFilePath, numImages);

    int input_layer = 784;
    int output_layer = 10;
    int number_hidden_layers = 3;
    int number_node_per_hidden = 512;

    for (const auto& entry : directory_iterator(DIR)) {
        filePaths.push(entry.path().filename().string());
    }

    unordered_map<string, Point_System> Points;

    while(!filePaths.empty())
    {
        string filePath = filePaths.front();
        filePaths.pop();
        NeuralNetwork network(input_layer, number_hidden_layers, number_node_per_hidden,
                  output_layer, 1.0,DIR+"3.453878/Network.bin", 105, true);

        // Reset network statistics before evaluation
        network.resetStatistics();

        // Run evaluation
        for(int i = 0; i < 100; i++) {
            vector<double> correct_label_output(10, 0.0);
            correct_label_output[labels[i]] = 1.0;
            network.testNetwork(images[i], correct_label_output);
        }

        // Get final statistics after all runs
        double final_cost = network.getCost();  // New method to get properly averaged cost
        Points.emplace(filePath, Point_System(network.precise_correct_count,
                                            network.vauge_correct_count,
                                            final_cost));
        cout << network.precise_correct_count <<endl;
    }

    for(auto net : Points) {
        if(net.second >= 0) {cout << "File Name :" << net.first << " Point Break Down :" << net.second << "\n";}
        }
        return 0;
    }
