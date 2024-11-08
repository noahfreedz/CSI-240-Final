#include <iostream>
#include <cstdlib>
#include <ctime>
#include "neural_network.h"
#include "window.h"
#include <SFML/Graphics.hpp>

using namespace std;

int main() {
    cout << "STARTING MAIN" << endl;

    NeuralNetwork myNN(784, 2, 16, 10); // 784 input nodes for 28x28 images

    Visualizer visualizer(800, 600, "SFML Window Example", myNN);
    visualizer.run();
}

