#include <iostream>
#include <cstdlib>
#include <ctime>
#include "neural_network.h"
#include "window.h"
#include <SFML/Graphics.hpp>

using namespace std;

int main() {
    cout << "STARTING MAIN" << endl;

    NeuralNetwork myNN(3, 3, 5, 3);

    Visualizer visualizer(800, 600, "SFML Window Example", myNN);
    visualizer.run();
}