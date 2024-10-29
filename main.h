#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <SFML/Graphics.hpp>

using namespace std;

double getRandom(double min, double max);
double LeakyReLU(double x, float bias);

class Node {
    public:
        static int last_layer;
        static int next_ID;
        int ID;
        int layer;
        float bias;
        float activation_value;

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

        // Node Connection Struct
        struct connection {
            Node* start_address;
            Node* end_address;
            int start_id;
            int end_id;
            double weight;
        };
        vector<connection> forward_connections;
        vector<connection> backward_connections;

        void setActivationValue(float x) {
            if(layer != 0) {
                cout << "| ERROR - EDITING ACTIVATION VALUE OF NON-INPUT ("<< layer << ") NODE!" << endl;
                return; // Prevent setting non-input node activation values
            }
            activation_value = x;
        }

        // Calculate Activation Value for Node
        void calculate_node() {
            double connection_total = 0;
            for (auto & connection : backward_connections) {
                // Add the activation value and the connection weight
                connection_total += connection.start_address->activation_value * connection.weight;
            }
            if(layer > 0 && layer < last_layer) {
                // Use Leaky Rectified Linear Unit for Hidden Layers
                activation_value = LeakyReLU(connection_total, bias);
            } else if (layer == last_layer) {
                // Otherwise use TanH for output
                activation_value = tanh(connection_total + bias);
            } else {
                cout << "| ERROR - NODE LAYER IS OUTSIDE NEURAL NETWORK" << endl;
            }
        }
};

// Initialize Static Node Variables
int Node::last_layer = 0;
int Node::next_ID = 0;

class NeuralNetwork {
    public:
        vector<Node> allNodes;

        NeuralNetwork(int iNode_count=1, int hLayer_count=1, int hNode_count=1, int oNode_count=1) {

            // Assign Variables
            input_node_count = iNode_count;
            hidden_layer_count = hLayer_count;
            hidden_node_count = hNode_count;
            output_node_count = oNode_count;

            // Randomize
            srand(static_cast<unsigned int>(time(nullptr)));

            // Input Layers
            for(int n=0; n<iNode_count; n++) {
                Node newInputNode(0);
                allNodes.push_back(newInputNode);
            }
            Node::last_layer += 1;

            // Hidden Layer
            for (int l = 0; l < hLayer_count; l++) {
                for (int n = 0; n < hNode_count; n++) {
                    Node newHiddenNode(Node::last_layer);
                    allNodes.push_back(newHiddenNode);
                }
                Node::last_layer += 1;
            }

            // Output Layer
            for(int n=0; n<oNode_count; n++) {
                Node newOutputNode(Node::last_layer);
                allNodes.push_back(newOutputNode);
            }

            cout << "THIS NN HAS " << Node::last_layer << "(" << Node::last_layer+1 << ")" << " LAYERS" << endl;

            // Make Connections
            int current_layer = 0;
            while (current_layer < Node::last_layer) {
                //cout << "MAKING CONNECTIONS FOR LAYER " << current_layer << endl;
                vector<Node*> startNodes;
                vector<Node*> endNodes;

                // Collect pointers to nodes in the current and next layer
                for (auto& node : allNodes) {
                    //cout << "Checking Node: " << node.ID << endl;
                    int layer = node.layer;
                    if (layer == current_layer) {
                        startNodes.push_back(&node);
                    } else if (layer == current_layer + 1) {
                        endNodes.push_back(&node);
                    }
                }

                for (auto* start_node : startNodes) {
                    for (auto* end_node : endNodes) {
                        Node::connection new_connection{};
                        new_connection.start_address = start_node;
                        new_connection.end_address = end_node;
                        new_connection.start_id = start_node->ID;
                        new_connection.end_id = end_node->ID;
                        new_connection.weight = getRandom(-1.0, 1.0);

                        // Add connection to nodes' lists
                        start_node->forward_connections.push_back(new_connection);
                        end_node->backward_connections.push_back(new_connection);
                    }
                }

                current_layer++;
            }

            // Print Out All Connections
            for (auto& node : allNodes) {
                //cout << "NODE ID: " << node.ID << ", LAYER: " << node.layer << ", BIAS: " << node.bias << endl;
                for (const auto& connection : node.forward_connections) {
                    //cout << "  Forward Connection to Node ID: " << connection.end_id
                    //     << " with initial weight: " << connection.weight << endl;
                }
                for (const auto& connection : node.backward_connections) {
                    //cout << "  Backward Connection from Node ID: " << connection.start_id
                    //     << " with initial weight: " << connection.weight << endl;
                }
            }
        }

        void run_network(vector<float> inputs) {
            cout << "RUNNING NETWORK" << endl;

            // Set Activations of Input Layer
            //cout << "NODE INPUTS: " << input_node_count << endl;

            int inputIndex = 0;
            for (auto& node : allNodes) {
                if (node.layer == 0) {
                    node.setActivationValue(inputs[inputIndex]);
                    //cout << "NODE ID: " << node.ID << " SET TO " << inputs[inputIndex] << endl;
                    inputIndex++;
                }
            }

            // Starting with 2nd Layer, Calculate Activations
            int current_layer = 1;
            //cout << " -- CALCULATING ACTIVATIONS -- " << endl;
            while (current_layer <= Node::last_layer) {
                //cout << "CALCULATING LAYER " << current_layer << endl;
                for (auto &node: allNodes) {
                    int layer = node.layer;
                    if (layer == current_layer) {
                        node.calculate_node();
                    }
                }
                current_layer++;
            }
            for(auto &node : allNodes) {
                if(node.layer == Node::last_layer) {
                    cout << "OUTPUT OF NODE " << node.ID << ": " << node.activation_value << endl;
                }
            }
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


        int input_node_count;
        int hidden_layer_count;
        int hidden_node_count; //per layer
        int output_node_count;
};

double getRandom(double min, double max) {
    return min + (static_cast<double>(rand()) / RAND_MAX) * (max - min);
}

double LeakyReLU(double x, float bias) {
    if(x > bias) {
        return x;
    } else {
        return x*0.01f;
    }
}

//NEXT
//save start and end of each layer
//so when iterating we can save time
//
//might not be worth it but who knows