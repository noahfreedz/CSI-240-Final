#include <iostream>
#include <cstdlib>
#include <ctime>
#include "neural_network.h"
#include <vector>
//#include "window.h"
//#include <SFML/Graphics.hpp>

using namespace std;

vector<pair<double, int>>average(vector<vector<pair<int, double>>>& _vector)
{
    vector<pair<double, int>> averagedWeights; // Start with an empty vector

    int count = 0;
    for (auto& base : _vector)
    {
        count++;
        for (auto& pair : base)
        {
            if(!averagedWeights[pair.first].first)
            {
                averagedWeights[pair.first].first = 0.0;
            }
            averagedWeights[pair.first].first += pair.second;
        }
    }

    for (auto& weight : averagedWeights)
    {
        weight.first = weight.first / count;
    }
    return averagedWeights;
}
int main() {
    cout << "STARTING MAIN" << endl;
    vector<double> values;
    vector<vector<pair<int, double>>> allWeights;


    NeuralNetwork myNN(8, 2, 5, 3);
    for(int i = 0; i < 10; i++)
    {
        for(int i = 0; i < 8; i++)
        {
            values.push_back(getRandom(0,1.0));
        }

        myNN.run_network(values);
        vector<pair<int, double>> neWeights = myNN.backPropigation(getRandom(1,3));
        allWeights.emplace_back(neWeights);
        vector<pair<double, int>> averagedWeights = average(allWeights);
        myNN.assignValues(averagedWeights);
        allWeights.clear();
        averagedWeights.clear();
    }










    //Visualizer visualizer(800, 600, "SFML Window Example", myNN);
    //visualizer.run();
}