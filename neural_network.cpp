#include <iostream>
#include <cstdlib>
#include <ctime>
#include "neural_network.h"
#include <vector>
#include <unordered_map>
//#include "window.h"
//#include <SFML/Graphics.hpp>

using namespace std;

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
    cout << "STARTING MAIN" << endl;
    vector<double> values;
    vector<unordered_map<int, double>> allWeights;


    NeuralNetwork myNN(8, 2, 5, 3);
    for(int i = 0; i < 10; i++)
    {
        for(int i = 0; i < 8; i++)
        {
            values.push_back(getRandom(0,1.0));
        }
        myNN.run_network(values);
        unordered_map<int, double> newWeights = myNN.backPropigation(getRandom(1,3));
        allWeights.emplace_back(newWeights);
        unordered_map<int, double> averagedWeights = average(allWeights);
        myNN.assignValues(averagedWeights);
        allWeights.clear();
        averagedWeights.clear();
    }










    //Visualizer visualizer(800, 600, "SFML Window Example", myNN);
    //visualizer.run();
}