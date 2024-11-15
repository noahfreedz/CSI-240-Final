#include "neural_network.h"

using namespace Rebecca;

mutex data_mutex;

int NeuralNetwork::nextID = 0;

Node::Node(int node_layer, int& nextID, double _bais) : activation_value(0.0), layer(node_layer) {
            ID = nextID;
            nextID++;
            bias = _bais;
        }

Node:: Node(int node_layer, int& nextID) : activation_value(0.0), layer(node_layer) {
            ID = nextID;
            nextID++;
            bias = getRandom(-15, 15);
        }

void Node:: setActivationValue(double x) {
            if (layer != 0) {
                cout << "| ERROR - EDITING ACTIVATION VALUE OF NON-INPUT (" << layer << ") NODE!" << endl;
                return;
            }
            activation_value = x;
        }

void  Node::calculate_node() {
            double connection_total = 0;
            if (layer != 0) {
                for (const auto& pair : backward_connections) {
                    connection_total += pair.second->start_address->activation_value * pair.second->weight;
                }
                activation_value = sigmoid(connection_total - bias);
            }
        }

NeuralNetwork:: NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count, double _learning_rate)
                    : next_ID(0), connection_ID(0), last_layer(0), ID(nextID++) {

                learning_rate = _learning_rate;

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

NeuralNetwork::NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count, double _learning_rate,
                vector<double>_startingWeights, vector<double> _startingBiases)
                    : learning_rate(_learning_rate), next_ID(0), connection_ID(0), last_layer(0), ID(nextID++) {

                // string weight_file = "outputWeights.bin";
                // string bais_file = "outputBiases.bin";
                // auto weigths = loadData(weight_file);
                // cout << weigths[67] << endl;
                // Create input layer nodes
                for (int n = 0; n < iNode_count; n++) {
                    Node newInputNode(0, next_ID);
                    allNodes.push_back(newInputNode);
                }
                last_layer++;

                // Create hidden layers
                for (int l = 0; l < hLayer_count; l++) {
                    for (int n = 0; n < hNode_count; n++) {
                        Node newHiddenNode(l+1, next_ID, _startingBiases[n*(l+1)]);
                        allNodes.push_back(newHiddenNode);
                    }
                    last_layer++;
                }

                // Create output layer nodes
                for (int n = 0; n < oNode_count; n++) {
                    Node newOutputNode(last_layer, next_ID, _startingBiases[n+(hLayer_count*hNode_count)]);
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
                            new_connection.weight = _startingWeights[new_connection.ID];

                            allConnections[new_connection.ID] = new_connection;
                            start_node->forward_connections[start_node->forward_connections.size()] = &allConnections[new_connection.ID];
                            end_node->backward_connections[end_node->backward_connections.size()] = &allConnections[new_connection.ID];
                        }
                    }
                    current_layer++;
                }
            }

NeuralNetwork::~NeuralNetwork() {
                weights_.clear();
                biases_.clear();
                backprop_count = 0;
                string weight_file = "outputWeights" +to_string(ID) +".bin";
                string bais_file = "outputBiases.bin";

                auto weights_and_biases = getWeightsAndBiases();


                saveData(weights_and_biases.first, weight_file);
                saveData(weights_and_biases.second, bais_file);

            }

double NeuralNetwork:: run_network(vector<double> inputs, vector<double> correct_outputs) {
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
            bool vauge_network_correct = true;
            bool precise_network_correct = true;
            for(auto &node : allNodes) {
                if(node.layer == last_layer) {
                    // Calculate Correct
                    if(vauge_network_correct) {
                        if(correct_outputs[output_count] == 1.0) {
                            if(node.activation_value <= 0.9) {
                                vauge_network_correct = false;
                            }
                        } else if (correct_outputs[output_count] == 0.0) {
                            if(node.activation_value > 0.3) {
                                vauge_network_correct = false;
                            }
                        }
                    }

                    // Calculate Correct
                    if(precise_network_correct) {
                        if(correct_outputs[output_count] == 1.0) {
                            if(node.activation_value < 0.9) {
                                precise_network_correct = false;
                            }
                        }
                        else if (correct_outputs[output_count] == 0.0) {
                            if(node.activation_value > 0.1) {
                                precise_network_correct = false;
                            }
                        }
                        else {
                            cout << "OUTPUT NOT RECOGNIZED: " << correct_outputs[output_count] << endl;
                        }
                    }

                    // Calculate Target Value
                    double target_val = correct_outputs[output_count] - node.activation_value;

                    // Calculate Node Error Value
                    node.error_value = node.activation_value * (1 - node.activation_value) * (target_val);
                    total_error += abs(node.error_value);
                    output_count++;
                    cost += pow(target_val, 2);
                }
            }
            if(vauge_network_correct) {
                vauge_correct_count++;
            }
            if(precise_network_correct) {
                precise_correct_count++;
            }
            total_outputs++;
            average_cost.emplace_back(cost);
            return total_error;
        }

void NeuralNetwork:: backpropigate_network()
                {
                    // New Weights To Implement
                    unordered_map<int, double> newWeights;
                    unordered_map<int, double> newBaises;
                    // Learning Rate 1 for default
                    double learningRate = 0.05;

                    // Increment Networks Run Count
                    runs++;

                    // Loop Through Layers Starting with Second To Last Going Backward
                    for(int i =last_layer - 1; 0 < i ; i--)
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
                        double nodeError = connection.second.end_address->error_value;
                        double weightChange =  this->learning_rate * nodeError * connection.second.start_address->activation_value;
                        double weightValue = connection.second.weight + weightChange;
                        newWeights[connection.second.ID] = weightValue;
                    }

                    for (auto& node : allNodes) {
                        if (node.layer != 0) {
                            // Update bias using the node's own error value
                            double biasValue = node.bias - this->learning_rate * node.error_value;
                            newBaises[node.ID] = biasValue;
                        }
                    }

                    weights_.push_back(newWeights);
                    biases_.push_back(newBaises);

                    backprop_count++;

                    if(backprop_count == upper_backprop_count) {
                        edit_weights(average(weights_));
                        edit_biases(average(biases_));
                    }
                }

pair< unordered_map<int, double>, unordered_map<int, double>>  NeuralNetwork:: getWeightsAndBiases() {
                unordered_map<int, double> newWeights;
                unordered_map<int, double> newBaises;
                for (auto connection : allConnections) {
                    // Calculate weight change
                    newWeights[connection.first] = connection.second.weight;
                }
                for (auto& node : allNodes) {
                    if (node.layer != 0) {
                        newBaises[node.ID] = node.bias;
                    }
                }
                return make_pair(newWeights, newBaises);
            }

double NeuralNetwork:: getCost() {
                double total_cost = 0.0;
                double endValue;
                int count = 0;
                for(auto cost: average_cost) {
                    total_cost += cost;
                    count++;
                }
                average_cost.clear();
                endValue = total_cost/count;
                return endValue;
            }

double NeuralNetwork:: getLearningRate() {
                return learning_rate;
            }

void  NeuralNetwork::edit_weights(const unordered_map<int, double>& new_values)
                {
                    for (auto& connection : allConnections)
                    {
                         connection.second.weight = new_values.at(connection.first);
                    }
                }

void  NeuralNetwork::edit_biases(const unordered_map<int, double>& new_biases)
                {
                    for(auto node: allNodes){
                        if(node.layer != 0)
                        {
                            node.bias = new_biases.at(node.ID);
                        }
                    }
                }

void  NeuralNetwork::saveData(const unordered_map<int, double>& data, const string& filename)
            {
                ofstream outFile(filename, ios::binary);
                if (!outFile) {
                    cerr << "Error opening file for writing\n";
                    return;
                }

                // Write the size of the vector first
                size_t dataSize = data.size();
                outFile.write(reinterpret_cast<const char*>(&dataSize), sizeof(dataSize));

                // Write each pair of ID and double value
                for (const auto& pair : data) {
                    outFile.write(reinterpret_cast<const char*>(&pair.first), sizeof(pair.first));
                    outFile.write(reinterpret_cast<const char*>(&pair.second), sizeof(pair.second));
                }

                outFile.close();
            }

unordered_map<int, double>  NeuralNetwork::loadData(const string& filename) {
                unordered_map<int, double> data;
                ifstream inFile(filename, ios::binary);
                if (!inFile) {
                    cerr << "Error: Could not open file for reading\n";
                    return data;
                }

                // Read the size of the map (originally the vector size)
                size_t dataSize;
                inFile.read(reinterpret_cast<char*>(&dataSize), sizeof(dataSize));

                // Read each pair of ID and double value and insert it into the map
                for (size_t i = 0; i < dataSize; ++i) {
                    int id;
                    double value;
                    inFile.read(reinterpret_cast<char*>(&id), sizeof(id));
                    inFile.read(reinterpret_cast<char*>(&value), sizeof(value));
                    data[id] = value;  // Insert into the unordered_map
                }

                inFile.close();
                return data;
            }

ThreadNetworks::ThreadNetworks(int number_networks, double lower_learning_rate,
               double upper_learning_rate, vector<double>& _startingWeights,
               vector<double>& _startingBiases, int input_node_count,
               int hidden_layer_count_, int node_per_hidden_layer, int output_node_count) {

                networks_.reserve(number_networks);
                double learning_rate_step = abs((upper_learning_rate - lower_learning_rate) / (number_networks-1));
                for (int i = 0; i < number_networks; i++) {
                    double current_learning_rate = lower_learning_rate + (i * learning_rate_step);

                    networks_.push_back(make_unique<NeuralNetwork>(
                            input_node_count, hidden_layer_count_, node_per_hidden_layer,
                            output_node_count, current_learning_rate, _startingWeights, _startingBiases));
                }

            }

void ThreadNetworks:: SetWindow(GraphWindow &window) {
    window_ = &window;
    for(auto& network : networks_) {
        window_->setLearningRate(network->ID, network->getLearningRate());
    }


}

void ThreadNetworks::runThreading(vector<double>& image, vector<double>& correct_label_output) {
                    vector<thread> threads;
                for (auto& network : networks_) {
                    threads.emplace_back([this, &network, &image, &correct_label_output]() {
                        trainNetwork(*network, image, correct_label_output);
                    });
                }
                // Join each thread
                for (auto& t : threads) {
                    t.join();
                }
            }

void ThreadNetworks::trainNetwork(NeuralNetwork& network, const vector<double>& input, const vector<double>& correct_output) {

                    network.run_network(input, correct_output);

                    // Lock data access to prevent race conditions during weight updates
                    lock_guard< mutex> guard(data_mutex);

                    network.backpropigate_network();
                }

void ThreadNetworks::PrintCost() {
                for(auto& network : networks_) {
                    window_->addDataPoint(network->ID,network->getCost());
                    cout << network->getLearningRate() << ": (VAUGE) " << network->vauge_correct_count << "/100 (PRECISE) " << network-> precise_correct_count << "/100" << endl;
                    network->vauge_correct_count = 0;
                    network->precise_correct_count = 0;
                }
            }

void ThreadNetworks::render() {
                window_->render();
                window_->handleEvents();
            }

void ThreadNetworks:: deleteNetworks() {
                cout << "Deleting networks" << endl;
            }