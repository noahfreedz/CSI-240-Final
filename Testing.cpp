
#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <thread>
#include <fstream>

#include "neural_network.h"

using namespace std;
using  namespace Rebecca;

string DIR = "../network/Best";

int main() {

    auto startingWights = loadData(DIR + "outputWeights4.bin");
    auto startingBais = loadData(DIR + "outputBiases4.bin");

    int input_layer = 784;
    int output_layer = 10;
    int number_hidden_layers = 2;
    int number_node_per_hidden = 128;

    NeuralNetwork testingNetwork(input_layer, number_hidden_layers,number_node_per_hidden,
        output_layer, .01, startingWights, startingBais );

    int count = 0;
    while (true) {
        count++;
    }

    return 0;
}