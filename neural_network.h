#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <chrono>
#include <thread>
#include <unordered_map>
#include <SFML/Graphics.hpp>

using namespace std;

double getRandom(double min, double max);
double sigmoid(double x) {
    double result = 1 / (1 + exp(-x));

    if (result < 0.000000001) {
        result = 0.0; // Set the minimum value
    }
    return result;
}
class Node;

struct connection {
    Node* start_address;
    Node* end_address;
    double weight;
    int ID;
    static int nextID;
};

class Node {
public:
    static int last_layer;
    static int next_ID;
    int ID;
    int layer;
    double bias;
    double activation_value;
    double error_value = 0;

    // Constructor
    Node(int node_layer) : activation_value(0.0f){
        // Define Node Type
        layer = node_layer;
        // Give Unique ID
        ID = next_ID;
        next_ID++;
        // Set Random Bias
        bias = getRandom(-15.0, 15.0);

    }

    // Node Connection Hashtable
    unordered_map<int , connection*> forward_connections;
    unordered_map<int , connection*> backward_connections;

    void setActivationValue(double x) {
        if(layer != 0) {
            cout << "| ERROR - EDITING ACTIVATION VALUE OF NON-INPUT ("<< layer << ") NODE!" << endl;
            return; // Prevent setting non-input node activation values
        }
        activation_value = x;
    }

    // Calculate Activation Value for Node
    void calculate_node() {
        double connection_total = 0;
        if(layer != 0)
        {for (const auto& pair : backward_connections) {
                //std::cout << pair.second->start_address->ID << ": " << pair.second->weight << std::endl;
                connection_total += pair.second->start_address->activation_value * pair.second->weight;
            }
        }

//
        if(layer > 0) {
            activation_value = sigmoid(connection_total-bias);
        }
        else {
            cout << "| ERROR - NODE LAYER IS OUTSIDE NEURAL NETWORK" << endl;
        }
    }
};


// Initialize Static Node Variables
int Node::last_layer = 0;
int Node::next_ID = 0;
int connection::nextID = 0;

class NeuralNetwork {
public:
    vector<Node> allNodes;
    unordered_map<int, connection> allConnections;
    vector<unordered_map<int, double>> all_weights;
    vector<unordered_map<int, double>> all_biases;
    vector<double> averageCost;

    NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count) {
        // Reserve For No Resizing
        allConnections.reserve(999999);

        // Input Layer Nodes
        for(int n=0; n<iNode_count; n++) {
            Node newInputNode(0);
            allNodes.push_back(newInputNode);
        }
        Node::last_layer += 1;

        // Hidden Layer(s) Nodes
        for (int l = 0; l < hLayer_count; l++) {
            for (int n = 0; n < hNode_count; n++) {
                Node newHiddenNode(Node::last_layer);
                allNodes.push_back(newHiddenNode);
            }
            Node::last_layer += 1;
        }

        // Output Layer Nodes
        for(int n=0; n<oNode_count; n++) {
            Node newOutputNode(Node::last_layer);
            allNodes.push_back(newOutputNode);
        }

        // Connections
        int current_layer = 0;
        while (current_layer < Node::last_layer) {
            vector<Node*> start_nodes;
            vector<Node*> end_nodes;

            // Collect pointers to nodes in the current and next layer
            for (auto& node : allNodes) {
                int layer = node.layer;
                if (layer == current_layer) {
                    start_nodes.push_back(&node);
                } else if (layer == current_layer + 1) {
                    end_nodes.push_back(&node);
                }
            }

            // Make Connections For All Nodes
            for (auto& start_node : start_nodes)
            {
                for (auto& end_node : end_nodes) {
                    // Initialize Connection
                    connection new_connection{};

                    // Unique ID per Connection
                    new_connection.ID = connection::nextID;
                    connection::nextID++;

                    // Initialize Connection Values
                    new_connection.start_address = start_node;
                    new_connection.end_address = end_node;
                    new_connection.weight = getRandom(-1.0, 1.0);

                    // Add to All Connections Hash
                    allConnections[new_connection.ID] = new_connection;

                    // Append to Forward Connections Of Start Node
                    int start_size = start_node->forward_connections.size();
                    start_node->forward_connections[start_size] = & allConnections[new_connection.ID];
                    // Append to Backward Connections Of End Node
                    int end_size = end_node->backward_connections.size();
                    end_node->backward_connections[end_size] = & allConnections[new_connection.ID];
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
        while (current_layer <= Node::last_layer) {
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
            if(node.layer == Node::last_layer) {
                // Calculate Target Value
                double target_val = correct_outputs[output_count] - node.activation_value;
                // Calculate Node Error Value
                node.error_value = node.activation_value * (1 - node.activation_value) * (target_val);
                total_error += std::abs(node.error_value);
                output_count++;
                cost += pow(target_val, 2);
            }
        }
        averageCost.emplace_back(cost);
        //cout << "NETWORK RUN (" << runs << ")" << " - TOTAL ERROR: " << total_error << endl;
        return total_error;
    }

    void backpropigate_network()
    {
        // New Weights To Implement
        unordered_map<int, double> newWeights;
        unordered_map<int, double> newBaises;
        // Learning Rate 1 for default
        double learningRate = 0.05;

        // Increment Networks Run Count
        runs++;
        internal_runs++;

        // Loop Through Layers Starting with Second To Last Going Backward
        for(int i = Node::last_layer - 1; 0 < i ; i--)
        {
            for(auto& node: allNodes) {
                if (node.layer == i && node.layer != Node::last_layer){
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
            double nodeError = connection.second.end_address->error_value;
            double weightChange = learningRate * nodeError * connection.second.start_address->activation_value;
            double weightValue = connection.second.weight + weightChange;
            newWeights[connection.second.ID] = weightValue;
        }

        for (auto& node : allNodes) {
            if (node.layer != 0) {
                // Update bias using the node's own error value
                double biasValue = node.bias - learningRate * node.error_value;
                newBaises[node.ID] = biasValue;
            }
        }

        all_weights.emplace_back(newWeights);
        all_biases.emplace_back(newBaises);
        if(internal_runs == betwenn_backprop) {
            unordered_map<int, double> averaged_weights = average(all_weights);
            unordered_map<int, double> averaged_biases =average(all_biases);
            edit_weights(averaged_weights);
            edit_biases(averaged_biases);
            internal_runs = 0;
            this_thread::sleep_for(chrono::milliseconds(1000));
        }
    }

    double getCost() {
        double total_cost = 0.0;
        double endValue;
        int count = 0;
        for(auto cost: averageCost) {
            total_cost += cost;
            count++;
        }
        endValue = total_cost/count;
        return endValue;



    }

    int nodes_in_layer(int layer) {
        int count = 0;
        for(auto &node : allNodes) {
            if (node.layer == layer) {
                count++;
            }
        }
        return count;
    }

private:
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

    int runs = 0;
    int correct = 0;
    int internal_runs = 0;
    int betwenn_backprop = 100;
};

double getRandom(double min, double max) {
    static std::random_device rd;                            // Seed source
    static std::mt19937 gen(rd());                      // Mersenne Twister generator initialized with random seed
    std::uniform_real_distribution<> dis(min, max);   // Distribution in the range [min, max]
    return dis(gen);                                     // Return Random
}