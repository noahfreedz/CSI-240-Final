#include <iostream>
#include <cstdlib>
#include <ctime>
#include "main.h"

using namespace std;

int main() {
    cout << "STARTING MAIN" << endl;

    NeuralNetwork myNN(3, 2, 2, 2);

    vector<float> inputs;
    for(int i = 0; i < 3; i++) {
        inputs.push_back((static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0);
    }
    myNN.run_network(inputs);
}