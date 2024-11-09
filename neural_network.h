#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
#include <list>
#include <unordered_map>
//#include <SFML/Graphics.hpp>

using namespace std;

double getRandom(double min, double max);
double sigmoid(double x);

class Node;

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
    Node(int node_layer, int& nextID) : activation_value(0.0), layer(node_layer) {
        ID = nextID;
        nextID++;
        bias = getRandom(-15.0, 15.0);
    }

    unordered_map<int, connection*> forward_connections;
    unordered_map<int, connection*> backward_connections;

    void setActivationValue(double x) {
        if (layer != 0) {
            cout << "| ERROR - EDITING ACTIVATION VALUE OF NON-INPUT (" << layer << ") NODE!" << endl;
            return;
        }
        activation_value = x;
    }

    void calculate_node() {
        double connection_total = 0;
        if (layer != 0) {
            for (const auto& pair : backward_connections) {
                connection_total += pair.second->start_address->activation_value * pair.second->weight;
            }
            activation_value = sigmoid(connection_total - bias);
        }
    }
};

class NeuralNetwork {
public:
    vector<Node> allNodes;
    unordered_map<int, connection> allConnections;
    vector<double> average_cost;
    int runs = 0;
    int correct = 0;
    double learning_rate = 0.05;

    // Instance-specific ID counters
    int next_ID;  // For nodes
    int connection_ID;  // For connections
    int last_layer;

    NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count, double _learning_rate = 0.05)
            : learning_rate(_learning_rate), next_ID(0), connection_ID(0), last_layer(0) {

        // Create input layer nodes
        for (int n = 0; n < iNode_count; n++) {
            Node newInputNode(0, next_ID);
            allNodes.push_back(newInputNode);
        }
        last_layer++;

        // Create hidden layers
        for (int l = 0; l < hLayer_count; l++) {
            for (int n = 0; n < hNode_count; n++) {
                Node newHiddenNode(last_layer, next_ID);
                allNodes.push_back(newHiddenNode);
            }
            last_layer++;
        }

        // Create output layer nodes
        for (int n = 0; n < oNode_count; n++) {
            Node newOutputNode(last_layer, next_ID);
            allNodes.push_back(newOutputNode);
        }

        // Create connections
        int current_layer = 0;
        while (current_layer < last_layer) {
            vector<Node*> start_nodes;
            vector<Node*> end_nodes;

            // Collect nodes in the current and next layer
            for (auto& node : allNodes) {
                if (node.layer == current_layer) {
                    start_nodes.push_back(&node);
                } else if (node.layer == current_layer + 1) {
                    end_nodes.push_back(&node);
                }
            }

            // Connect nodes between layers
            for (auto& start_node : start_nodes) {
                for (auto& end_node : end_nodes) {
                    connection new_connection{};
                    new_connection.ID = connection_ID;
                    connection_ID++;
                    new_connection.start_address = start_node;
                    new_connection.end_address = end_node;
                    new_connection.weight = getRandom(-1.0, 1.0);

                    allConnections[new_connection.ID] = new_connection;
                    start_node->forward_connections[start_node->forward_connections.size()] = &allConnections[new_connection.ID];
                    end_node->backward_connections[end_node->backward_connections.size()] = &allConnections[new_connection.ID];
                }
            }
            current_layer++;
        }
    }


        double run_network(vector<double> inputs, vector<double> correct_outputs) {
            int inputIndex = 0;
            for (auto& node : allNodes) {
                if (node.layer == 0) {
                    node.setActivationValue(inputs[inputIndex]);
                    inputIndex++;
                }
            }

            // Starting with 2nd Layer, Calculate Activations
            int current_layer = 1;
            while (current_layer <= last_layer) {
                for (auto &node: allNodes) {
                    int layer = node.layer;
                    if (layer == current_layer) {
                        node.calculate_node();
                    }
                }
                current_layer++;
            }

            // Calculate Error for all Output Nodes
            int output_count = 0;
            double total_error = 0.0;
            double cost = 0.0;
            for(auto &node : allNodes) {
                if(node.layer == last_layer) {
                    // Calculate Target Value
                    double target_val = correct_outputs[output_count] - node.activation_value;
                    // Calculate Node Error Value
                    node.error_value = node.activation_value * (1 - node.activation_value) * (target_val);
                    total_error += std::abs(node.error_value);
                    output_count++;
                    cost += pow(target_val, 2);
                }
            }
            average_cost.emplace_back(cost);
            return total_error;
        }

        pair<unordered_map<int, double>, unordered_map<int, double>> backpropigate_network()
            {
                // New Weights To Implement
                unordered_map<int, double> new_weights;
                unordered_map<int, double> new_biases;
                // Increment Networks Run Count
                runs++;

                // Loop Through Layers Starting with Second To Last Going Backward
                for(int i = last_layer - 1; 0 < i ; i--)
                {
                    for(auto& node: allNodes) {
                        if (node.layer == i && node.layer != last_layer){
                            // Sum Error Values of Previous Layer to Get Error From Current Node
                            double error_sum = 0;
                            for(auto _connection : node.forward_connections) {
                                error_sum += _connection.second->weight * _connection.second->end_address->error_value;
                            }
                            // Set Error Value
                            node.error_value = node.activation_value*(1-node.activation_value)*(error_sum);
                        }
                    }
                }

                // Determine New Weights for All Connections and Biases for each node
                for (auto connection : allConnections) {
                    // Calculate weight change
                    double node_error = connection.second.end_address->error_value;
                    double weight_change = learning_rate * node_error * connection.second.start_address->activation_value;
                    double weight_value = connection.second.weight + weight_change;
                    new_weights[connection.second.ID] = weight_value;
                }

                for (auto& node : allNodes) {
                    if (node.layer != 0) {
                        // Update bias using the node's own error value
                        double biasValue = node.bias - learning_rate * node.error_value;
                        new_biases[node.ID] = biasValue;
                    }
                }

                return make_pair(new_weights, new_biases);
            }

        double get_cost() {
            if (average_cost.empty()) return 0.0; // Avoid division by zero

            double total_cost = 0.0;
            for (double cost : average_cost) {
                total_cost += cost;
            }

            double endValue = total_cost / average_cost.size();
            average_cost.clear(); // Clear after calculation

            return endValue;
        }


        void edit_weights(const unordered_map<int, double> new_values)
            {
                for (auto& connection : allConnections)
                {
                     connection.second.weight = new_values.at(connection.first);
                }
            }

        void edit_biases(const unordered_map<int, double> new_biases)
            {
                for(auto node: allNodes){
                    if(node.layer != 0)
                    {
                        node.bias = new_biases.at(node.ID);
                    }
                }
            }
};

double getRandom(double min, double max) {
    static std::random_device rd;                            // Seed source
    static std::mt19937 gen(rd());                      // Mersenne Twister generator initialized with random seed
    std::uniform_real_distribution<> dis(min, max);   // Distribution in the range [min, max]
    return dis(gen);                                     // Return Random
}

double sigmoid(double x) {
    double result = 1 / (1 + exp(-x));

    if (result < 0.000000001) {
        result = 0.0; // Set the minimum value
    }
    return result;
}