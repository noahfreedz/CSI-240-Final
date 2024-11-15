#pragma once
#include <iostream>
#include <map>
#include <vector>
#include <numeric>
#include <thread>
#include <mutex>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <random>
#include <unordered_map>
#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "SFML/Graphics/Font.hpp"
#include "SFML/Graphics/RenderWindow.hpp"
#include "SFML/Window/Event.hpp"
#include "SFML/Graphics/RectangleShape.hpp"
#include "SFML/Graphics/Text.hpp"
#include "SFML/Graphics/CircleShape.hpp"


using namespace std;

namespace Rebecca {

    class ThreadNetworks;
    class Node;
    class NeuralNetwork;


    inline double getRandom(double min, double max) {
        static random_device rd;                            // Seed source
        static mt19937 gen(rd());                      // Mersenne Twister generator initialized with random seed
        uniform_real_distribution<> dis(min, max);   // Distribution in the range [min, max]
        return dis(gen);                                     // Return Random
    }

    inline double sigmoid(double x) {
        double result = 1 / (1 + exp(-x));

        if (result < 0.000000001) {
            result = 0.0; // Set the minimum value
        }
        return result;
    }

    inline unordered_map<int, double> average(vector<unordered_map<int, double>>& _vector)
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

    struct connection {
        Node* start_address;
        Node* end_address;
        double weight;
        int ID;  // Assigned by the NeuralNetwork instance
    };

    class Node {
    public:
        int ID;
        int layer;
        double bias;
        double activation_value;
        double error_value = 0;

        // Constructor
        Node(int node_layer, int& nextID, double _bais);
        Node(int node_layer, int& nextID);

        unordered_map<int, connection*> forward_connections;
        unordered_map<int, connection*> backward_connections;

        void setActivationValue(double x);

        void calculate_node();
    };

    class GraphWindow {
        public:
            bool run_network = true;

            GraphWindow(unsigned int width, unsigned int height, const string& title, ThreadNetworks* _allNetworks);

            bool isOpen() const;

            void handleEvents();

            void addDataPoint(int lineID, double value);

            void setLearningRate(int lineID, double learningRate);

            void render();

        private:

            sf::RenderWindow window;
            float window_width;
            float window_height;
            int perRunCount = 500;
            sf::Font font;
            ThreadNetworks* allNetworks;
            int mouseX = -1; // To track mouse X position
            std::map<int, std::vector<double>> dataSets;
            std::map<int, double> learningRates;
            std::map<int, sf::Color> colors;

            void drawGraph();

            void drawCursorLineWithMarkers();

            void drawAxesLabels();

            void drawAxisTitles();

            void drawKey();

            sf::Color generateColor(int index);

            void drawYAxisLabel(double value, float x, float y);

            void drawXAxisLabel(int run, float x, float y);

            std::string formatLabel(double value);
        };

    class NeuralNetwork {
        public:
            int ID;
            static int nextID;
            int vauge_correct_count = 0;
            int precise_correct_count = 0;
            int total_outputs = 0;

            //NeuralNetwork(){}
            NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count, double _learning_rate);

            NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count, double _learning_rate,
                vector<double>_startingWeights, vector<double> _startingBiases);

            NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count, double _learning_rate,
               vector<double>_startingWeights, vector<double> _startingBiases, string WeightFilePath, string BaisFilePath);

            ~NeuralNetwork();
            double run_network(vector<double> inputs, vector<double> correct_outputs);

            void backpropigate_network();

            pair< unordered_map<int, double>, unordered_map<int, double>> getWeightsAndBiases();

            double getCost();

            double getLearningRate();
    private:

            vector<Node> allNodes;
            unordered_map<int, connection> allConnections;
            vector<double> average_cost;
            vector<unordered_map<int, double>> weights_;
            vector<unordered_map<int, double>> biases_;

            int runs = 0;
            int correct = 0;
            double learning_rate;

            int backprop_count = 0;
            int upper_backprop_count = 100;

            // Instance-specific ID counters
            int next_ID;  // For nodes
            int connection_ID;  // For connections
            int last_layer;

            void edit_weights(const unordered_map<int, double>& new_values);

            void edit_biases(const unordered_map<int, double>& new_biases);

            void saveData(const unordered_map<int, double>& data, const string& filename);

            unordered_map<int, double> loadData(const string& filename);
    };

    class ThreadNetworks
    {
        public:
            ThreadNetworks(int number_networks, double lower_learning_rate,
               double upper_learning_rate, vector<double>& _startingWeights,
               vector<double>& _startingBiases, int input_node_count,
               int hidden_layer_count_, int node_per_hidden_layer, int output_node_count);

            void SetWindow(GraphWindow &window);

            void runThreading(vector<double>& image, vector<double>& correct_label_output);

            void trainNetwork(NeuralNetwork& network, const vector<double>& input, const vector<double>& correct_output);

            void PrintCost();

            void render();
            void deleteNetworks();


    private:
        int NetworkID = 0;
        int ThreadID = 0;

        GraphWindow* window_;

        vector<unique_ptr<NeuralNetwork>> networks_;
    };

}