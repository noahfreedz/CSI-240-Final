#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <random>
//#include <SFML/Graphics.hpp>

using namespace std;

double getRandom(double min, double max);
double sigmoid(double x) {
    return 1 / (1 + exp(-x));
}

class Node {
    public:
        static int last_layer;
        static int next_ID;
        int ID;
        int layer;
        double bias;
        double activation_value;
        double nodeError = 0;

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
            int ID;
            static int nextID;

            connection() : ID(nextID++) {}
        };

    vector<connection*> forward_connections; // Change to store pointers
    vector<connection*> backward_connections;

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
            for (auto & connection : backward_connections) {
                // Add the activation value and the connection weight
                connection_total += connection->start_address->activation_value * connection->weight;
            }
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
int Node::connection::nextID = 0;

class NeuralNetwork {
    public:
        vector<Node> allNodes;
        vector<Node::connection*> allConnections;

        NeuralNetwork(int iNode_count=1, int hLayer_count=1, int hNode_count=1, int oNode_count=1) {

            // Assign Variables
            input_node_count = iNode_count;
            hidden_layer_count = hLayer_count;
            hidden_node_count = hNode_count;
            output_node_count = oNode_count;

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
                        auto new_connection = new Node::connection(); // Use dynamic allocation
                        new_connection->start_address = start_node;
                        new_connection->end_address = end_node;
                        new_connection->start_id = start_node->ID;
                        new_connection->end_id = end_node->ID;
                        new_connection->weight = getRandom(-1.0, 1.0);

                        start_node->forward_connections.push_back(new_connection);
                        end_node->backward_connections.push_back(new_connection);
                        allConnections.push_back(new_connection);
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

        ~NeuralNetwork() {
            for (auto connection : allConnections) {
                delete connection; // Free the dynamically allocated memory
            }
        }

        void run_network(vector<double> inputs) {
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
        int getPositionInLayer(int nodeId, int layer)
        {
            int nodelayerCount = 0;
            for(auto node: allNodes)
            {
                if(node.layer == layer)
                {
                    nodelayerCount++;
                    if(node.ID == nodeId)
                    {
                        return nodelayerCount;
                    }
                }
            }

            return -1;

        }

        vector<pair<int, double>> backPropigation(int correctNode)
        {
            double learningRate = .5;
            vector<pair<double, Node>> activations;
            vector<pair<int, double>>newWeights;
            for(int i = Node::last_layer; 0 < i ; i--)
            {
                for(auto& upperNode: allNodes) {
                    if (upperNode.layer == i) {

                        //cout << "Test Print for each node Node Id: " << upperNode.ID << " Layer :" << i << endl;
                        double targetValue = 0.0;

                        // sets the
                        if(i == Node::last_layer)
                        {
                            int nodePos = getPositionInLayer(upperNode.ID, i);
                            if(nodePos == correctNode)
                            {
                                activations.push_back(make_pair(1.0, upperNode));
                            }
                            else
                            {
                                activations.push_back(make_pair(0.0, upperNode));
                            }

                        }

                        if(i == Node::last_layer)
                        {
                            for(const auto& targetNode: activations)
                            {
                                if(targetNode.second.ID == upperNode.ID)
                                {
                                    targetValue = targetNode.first - targetNode.second.activation_value;
                                }
                            }

                                upperNode.nodeError = upperNode.activation_value*(1-upperNode.activation_value)*(targetValue);
                                //cout << upperNode.nodeError << endl;
                        }

                        else
                        {
                            double sumErrorvaules = 0;
                            for(auto forwardConnect :upperNode.forward_connections)
                            {
                               sumErrorvaules += forwardConnect->weight * forwardConnect->end_address ->nodeError;

                            }
                            upperNode.nodeError = upperNode.activation_value*(1-upperNode.activation_value)*(sumErrorvaules);
                            //cout << upperNode.nodeError << endl;
                        }

                    }
                }

            }
            for (auto connection : allConnections) {
                double nodeError = connection->end_address->nodeError;
                double weightChange =   learningRate * nodeError * connection->start_address->activation_value;
                double tempValue = weightChange + connection->weight;
                newWeights.emplace_back(connection->ID, tempValue);
            }

            return newWeights;
        }

        void assignValues(const vector<double> averagedWeights)
        {


            for (auto& connection : allConnections)
            {
//                cout << "From connection " << connection.ID << " the weight has been changed from "
////                     << connection.weight << " to " << averagedWeights[connection.ID]
////                     << " that is a change of " << change << endl;

                connection->weight = averagedWeights[connection->ID];
            }
    }


private:
        int input_node_count;
        int hidden_layer_count;
        int hidden_node_count; //per layer
        int output_node_count;
};

double getRandom(double min, double max) {
    static std::random_device rd;          // Seed source
    static std::mt19937 gen(rd());         // Mersenne Twister generator initialized with random seed
    std::uniform_real_distribution<> dis(min, max); // Distribution in the range [min, max]
    return dis(gen);
}


//NEXT
//save start and end of each layer
//so when iterating we can save time
//
//might not be worth it but who knows