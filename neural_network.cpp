#include <iostream>
#include <cstdlib>
#include <ctime>
#include "neural_network.h"
#include <vector>
//#include "window.h"
//#include <SFML/Graphics.hpp>

using namespace std;

int main() {
    cout << "STARTING MAIN" << endl;
    vector<float> values;


    NeuralNetwork myNN(8, 5, 5, 3);
    for(float i = 0.1; i < 8.01; i++)
    {
       values.push_back(i);
    }

    myNN.run_network(values);
    myNN.backPropigation();

    //Visualizer visualizer(800, 600, "SFML Window Example", myNN);
    //visualizer.run();
}