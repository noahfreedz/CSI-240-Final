#include "neural_network.h"

using namespace Rebecca;

mutex data_mutex;

int NeuralNetwork::nextID = 0;

void debugNodeCalculation(const Node& node, double connection_total) {
    cout << "\nDetailed analysis of node " << node.ID << ":" << endl;
    cout << "Number of connections: " << node.backward_connections.size() << endl;

    double sum_pos = 0;
    double sum_neg = 0;
    int active_inputs = 0;

    for (const auto& pair : node.backward_connections) {
        if (pair.second->start_address->activation_value > 0) {
            active_inputs++;
            double contribution = pair.second->start_address->activation_value * pair.second->weight;
            if (contribution > 0) {
                sum_pos += contribution;
            } else {
                sum_neg += contribution;
            }
        }
    }

    cout << "Active inputs: " << active_inputs << endl;
    cout << "Sum of positive contributions: " << sum_pos << endl;
    cout << "Sum of negative contributions: " << sum_neg << endl;
    cout << "Total before bias: " << connection_total << endl;
    cout << "Bias: " << node.bias << endl;
    cout << "Final pre-ReLU value: " << (connection_total - node.bias) << endl;
}

Node::Node(int node_layer, int& nextID, double _bais) : activation_value(0.0), layer(node_layer) {
            ID = nextID;
            nextID++;
            bias = _bais;
        }

Node:: Node(int node_layer, int& nextID) : activation_value(0.0), layer(node_layer)
{
            ID = nextID;
            nextID++;
            if(layer != 0) {
                bias = getRandom(-1.0,1.0);
             }
            else {
                bias = 0;
            }
}

void Node:: setActivationValue(double x) {
            if (layer != 0) {
                cout << "| ERROR - EDITING ACTIVATION VALUE OF NON-INPUT (" << layer << ") NODE!" << endl;
                return;
            }
            activation_value = x;
        }

void Node::calculate_node() {
    if (layer != 0) {
        double connection_total = 0;
        for (const auto& pair : backward_connections) {
            connection_total += pair.second->start_address->activation_value *
                              pair.second->weight;
        }
        const double leaky_alpha = 0.2;
        if (layer == 3) {
            activation_value = sigmoid(connection_total);
        } else {
            activation_value = connection_total > 0 ?
                connection_total :
                (leaky_alpha * connection_total);
        }
    }
}

NeuralNetwork::NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count, double _learning_rate, int backprop_after )
                    : next_ID(0), connection_ID(0), last_layer(0), ID(nextID++), upper_backprop_count(backprop_after), SAVE_INTERVAL(backprop_after) {

                learning_rate = _learning_rate;
                // Create input layer nodes
                for (int n = 0; n < iNode_count; n++) {
                    Node newInputNode(0, next_ID);
                    allNodes.push_back(newInputNode);
                }
                last_layer++;

                // Create hidden layers
                int current_layer_nodes = hNode_count;  // Start with initial count (e.g. 512)
                const int MIN_NODES = 64;  // Don't go below this number
                for (int l = 0; l < hLayer_count; l++) {
                    for (int n = 0; n < current_layer_nodes; n++) {
                        Node newHiddenNode(last_layer, next_ID);
                        allNodes.push_back(newHiddenNode);
                    }
                    current_layer_nodes = max(MIN_NODES, current_layer_nodes / 2);  // Halve but don't go below minimum
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
                            new_connection.weight = getRandom(-2, 2);

                            allConnections[new_connection.ID] = new_connection;
                            start_node->forward_connections[start_node->forward_connections.size()] = &allConnections[new_connection.ID];
                            end_node->backward_connections[end_node->backward_connections.size()] = &allConnections[new_connection.ID];
                        }
                    }
                    current_layer++;
                }
            }

NeuralNetwork::NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count, double _learning_rate,
                vector<double>_startingWeights, vector<double> _startingBiases, int backprop_after)
                    : learning_rate(_learning_rate), next_ID(0), connection_ID(0), last_layer(0),  SAVE_INTERVAL(backprop_after), ID(nextID++), upper_backprop_count(backprop_after) {

                // Create input layer nodes
                for (int n = 0; n < iNode_count; n++) {
                    Node newInputNode(0, next_ID);
                    allNodes.push_back(newInputNode);
                }
                last_layer++;

                // Create hidden layers
                int current_layer_nodes = hNode_count;  // Start with initial count (e.g. 512)
                for (int l = 0; l < hLayer_count; l++) {
                    for (int n = 0; n < current_layer_nodes; n++) {
                        Node newHiddenNode(l+1, next_ID, _startingBiases[n*(l+1)]);
                        allNodes.push_back(newHiddenNode);
                    }
                    current_layer_nodes = current_layer_nodes /2;
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

NeuralNetwork::NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count,
    double _learning_rate,const string& FilePath, int backprop_after)
    :learning_rate(_learning_rate), next_ID(0), connection_ID(0), last_layer(0), ID(nextID++),upper_backprop_count(backprop_after), SAVE_INTERVAL(backprop_after)
{

    for (int n = 0; n < iNode_count; n++) {
        Node newInputNode(0, next_ID);
        allNodes.push_back(newInputNode);
    }
    last_layer++;

    // Create hidden layers
    int current_layer_nodes = hNode_count;  // Start with initial count (e.g. 512)
    for (int l = 0; l < hLayer_count; l++) {
        for (int n = 0; n < current_layer_nodes; n++) {
            Node newHiddenNode(l+1, next_ID, 0);
            allNodes.push_back(newHiddenNode);
        }
        current_layer_nodes = current_layer_nodes /2;
        last_layer++;
    }

    // Create output layer nodes
    for (int n = 0; n < oNode_count; n++) {
        Node newOutputNode(last_layer, next_ID, 0);
        allNodes.push_back(newOutputNode);
    }
    // Connections
    int current_layer = 0;
    while (current_layer < last_layer) {
        vector<Node*> start_nodes;
        vector<Node*> end_nodes;

        // Collect nodes in the current and next layers
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
                new_connection.ID = connection_ID++;
                new_connection.start_address = start_node;
                new_connection.end_address = end_node;
                new_connection.weight = 0; // Assign weight dynamically

                allConnections[new_connection.ID] = new_connection;
                start_node->forward_connections[start_node->forward_connections.size()] = &allConnections[new_connection.ID];
                end_node->backward_connections[end_node->backward_connections.size()] = &allConnections[new_connection.ID];
            }
        }
        current_layer++;
    }

    loadWeightsAndBiases(FilePath);
}

NeuralNetwork::NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count,
        double _learning_rate,const string& FilePath, int backprop_after, bool fileSorting)
    :learning_rate(_learning_rate), next_ID(0), connection_ID(0), SAVE_INTERVAL(backprop_after), last_layer(0), ID(nextID++),upper_backprop_count(backprop_after),
    fileSorting(fileSorting)
{

    for (int n = 0; n < iNode_count; n++) {
        Node newInputNode(0, next_ID);
        allNodes.push_back(newInputNode);
    }
    last_layer++;

    // Create hidden layers
    int current_layer_nodes = hNode_count;  // Start with initial count (e.g. 512)

    for (int l = 0; l < hLayer_count; l++) {
        for (int n = 0; n < current_layer_nodes; n++) {
            Node newHiddenNode(l+1, next_ID, 0);
            allNodes.push_back(newHiddenNode);
        }
        last_layer++;
    }

    // Create output layer nodes
    for (int n = 0; n < oNode_count; n++) {
        Node newOutputNode(last_layer, next_ID, 0);
        allNodes.push_back(newOutputNode);
    }
    // Connections
    int current_layer = 0;
    while (current_layer < last_layer) {
        vector<Node*> start_nodes;
        vector<Node*> end_nodes;

        // Collect nodes in the current and next layers
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
                new_connection.ID = connection_ID++;
                new_connection.start_address = start_node;
                new_connection.end_address = end_node;
                new_connection.weight = 0; // Assign weight dynamically

                allConnections[new_connection.ID] = new_connection;
                start_node->forward_connections[start_node->forward_connections.size()] = &allConnections[new_connection.ID];
                end_node->backward_connections[end_node->backward_connections.size()] = &allConnections[new_connection.ID];
            }
        }
        current_layer++;
    }

    loadWeightsAndBiases(FilePath);
}

NeuralNetwork::~NeuralNetwork()
{
    if(!fileSorting) {
        saveNetworkData();
    }
    clearConnections();
    clearNodes();
}

void NeuralNetwork::backpropigate_network(const vector<double>& inputs, const vector<double>& correct_outputs) {
    // Set input values
    int inputIndex = 0;
    for (auto& node : allNodes) {
        if (node.layer == 0) {
            node.setActivationValue(inputs[inputIndex]);
            inputIndex++;
        }
    }

    // Forward propagation
    int current_layer = 1;
    while (current_layer <= last_layer) {
        for (auto& node : allNodes) {
            if (node.layer == current_layer) {
                double connection_total = 0;
                for (const auto& pair : node.backward_connections) {
                    connection_total += pair.second->start_address->activation_value *
                                     pair.second->weight;
                }

                node.pre_activation = connection_total - node.bias;

                // Use ReLU for hidden layers
                if (current_layer != last_layer) {
                    node.activation_value = max(0.0, node.pre_activation);
                } else {
                    // Store raw values for softmax
                    node.activation_value = node.pre_activation;
                }
            }
        }

        // Apply softmax to output layer
        if (current_layer == last_layer) {
            // Compute softmax with numerical stability
            double max_val = -numeric_limits<double>::infinity();
            for (auto& node : allNodes) {
                if (node.layer == last_layer) {
                    max_val = max(max_val, node.activation_value);
                }
            }

            double sum_exp = 0.0;
            for (auto& node : allNodes) {
                if (node.layer == last_layer) {
                    node.activation_value = exp(node.activation_value - max_val);
                    sum_exp += node.activation_value;
                }
            }

            // Normalize
            for (auto& node : allNodes) {
                if (node.layer == last_layer) {
                    node.activation_value /= sum_exp;
                }
            }
        }
        current_layer++;
    }

    double total_cost = 0.0;
    int output_count = 0;

    for (auto& node : allNodes) {
        if (node.layer == last_layer) {
            double y_true = correct_outputs[output_count];
            double y_pred = max(1e-15, min(1.0 - 1e-15, node.activation_value));

            // CCE loss for this output
            if (y_true > 0) {
                total_cost -= y_true * log(y_pred);
            }

            // Gradient for softmax with CCE is simple: prediction - target
            node.error_value = y_pred - y_true;
            output_count++;
        }
    }

    // Store the cost (add this section)
    {
        lock_guard<mutex> lock(cost_mutex);
        if (isfinite(total_cost) && total_cost > 0) {
            average_cost.push_back(total_cost);
        }
    }

    // Add L2 regularization to cost if weight decay is enabled
    if (weight_decay > 0) {
        double l2_cost = 0.0;
        for (const auto& connection : allConnections) {
            l2_cost += 0.5 * weight_decay * pow(connection.second.weight, 2);
        }
        total_cost += l2_cost;
    }

    // Backpropagate through hidden layers
    for (int i = last_layer - 1; i > 0; i--) {
        for (auto& node : allNodes) {
            if (node.layer == i) {
                double error_sum = 0.0;
                for (const auto& connection : node.forward_connections) {
                    error_sum += connection.second->weight *
                                connection.second->end_address->error_value;
                }
                // ReLU derivative
                node.error_value = error_sum * (node.pre_activation > 0 ? 1.0 : 0.0);
            }
        }
    }

    // Update weights with improved gradient scaling
    const double scale_factor = 1.0 / sqrt(backprop_count + 1);

    unordered_map<int, double> newWeights;
    unordered_map<int, double> newBiases;

    for (auto& connection : allConnections) {
        double nodeError = connection.second.end_address->error_value;
        double activation = connection.second.start_address->activation_value;

        // Scale the learning rate based on layer depth
        double effective_lr = learning_rate * scale_factor;

        double raw_weight_change = effective_lr * nodeError * activation;
        raw_weight_change -= weight_decay * connection.second.weight;

        double weightChange = raw_weight_change;
        if (prev_weight_changes.find(connection.second.ID) != prev_weight_changes.end()) {
            weightChange += momentum * prev_weight_changes[connection.second.ID];
        }

        prev_weight_changes[connection.second.ID] = weightChange;

        // Gradient clipping
        weightChange = max(-0.1, min(0.1, weightChange));

        double new_weight = connection.second.weight + weightChange;
        newWeights[connection.second.ID] = new_weight;
    }

    // Update biases with similar improvements
    for (auto& node : allNodes) {
        if (node.layer != 0) {
            double effective_lr = learning_rate * scale_factor;
            double biasChange = effective_lr * node.error_value;

            if (prev_bias_changes.find(node.ID) != prev_bias_changes.end()) {
                biasChange += momentum * prev_bias_changes[node.ID];
            }

            prev_bias_changes[node.ID] = biasChange;

            // Gradient clipping for biases
            biasChange = max(-0.1, min(0.1, biasChange));

            double new_bias = node.bias + biasChange;
            newBiases[node.ID] = new_bias;
        }
    }

    weights_.push_back(newWeights);
    biases_.push_back(newBiases);

    if (++backprop_count == upper_backprop_count) {
        auto avgWeights = batchAverage(weights_);
        auto avgBiases = batchAverage(biases_);

        edit_weights(avgWeights);
        edit_biases(avgBiases);

        weights_.clear();
        biases_.clear();
        backprop_count = 0;

        // Adaptive learning rate decay
        learning_rate *= 0.995;  // Slower decay
        learning_rate = max(0.0001, learning_rate);  // Higher minimum
    }
}

void NeuralNetwork::resetStatistics() {
    lock_guard<mutex> lock(statsMutex);
    average_cost.clear();
    precise_correct_count = 0;
    vauge_correct_count = 0;
    guess_correct_count = 0;  // Reset the new counter
}

pair< unordered_map<int, double>, unordered_map<int, double>>  NeuralNetwork:: getWeightsAndBiases() {
                unordered_map<int, double> newWeights;
                unordered_map<int, double> newBaises;
                for (auto connection : allConnections) {
                    newWeights[connection.first] = connection.second.weight;
                }
                for (auto& node : allNodes) {
                    if (node.layer != 0) {
                        newBaises[node.ID] = node.bias;
                    }
                }
                return make_pair(newWeights, newBaises);
            }

double NeuralNetwork::getCost() {
    lock_guard<mutex> lock(cost_mutex);

    if (average_cost.empty()) {
        return current_cost;
    }

    // Calculate median instead of mean for more stability
    vector<double> valid_costs;
    for (double cost : average_cost) {
        if (isfinite(cost) && cost > 0) {
            valid_costs.push_back(cost);
        }
    }

    if (valid_costs.empty()) {
        return current_cost;
    }

    sort(valid_costs.begin(), valid_costs.end());
    size_t mid = valid_costs.size() / 2;

    if (valid_costs.size() % 2 == 0) {
        current_cost = (valid_costs[mid-1] + valid_costs[mid]) / 2.0;
    } else {
        current_cost = valid_costs[mid];
    }

    cost_sample_count = valid_costs.size();
    average_cost.clear();

    return current_cost;
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

void  NeuralNetwork::edit_biases(const unordered_map<int, double>& new_biases){
    for(auto node: allNodes) {
        if(node.layer != 0)
        {
            node.bias = new_biases.at(node.ID);
        }
    }
}

void NeuralNetwork:: saveNetworkData() {
    // Clear previous data
    weights_.clear();
    biases_.clear();
    backprop_count = 0;
    double cost = getCost();
    // Get current date and time
    auto now = chrono::system_clock::to_time_t(chrono::system_clock::now());
    tm local_time;
    localtime_s(&local_time, &now);

    // Format date and time
    stringstream ss;

    // Generate directory name with date, time, and cost
    string dir_name = "../network/" + to_string(cost);

    // Create the directory
    if (!exists(dir_name)) {
        create_directory(dir_name);
    }

    // Define file paths
    string weight_file = dir_name + "/Network.bin";

    // Get weights and biases
    auto weights_and_biases = getWeightsAndBiases();

    // Save weights and biases to the new directory
    saveWeightsAndBiases(weight_file);
}

void NeuralNetwork::saveWeightsAndBiases(const string& filename) {
    // Retrieve weights and biases
    auto [weights, biases] = getWeightsAndBiases();

    // Open the file for binary output
    std::ofstream outFile(filename, std::ios::binary);
    if (!outFile) {
        std::cerr << "Error: Could not open file for saving weights and biases: " << filename << std::endl;
        return;
    }

    // Save weights
    size_t weightCount = weights.size();
    outFile.write(reinterpret_cast<const char*>(&weightCount), sizeof(weightCount));
    for (const auto& [id, weight] : weights) {
        outFile.write(reinterpret_cast<const char*>(&id), sizeof(id));
        outFile.write(reinterpret_cast<const char*>(&weight), sizeof(weight));
    }

    // Save biases
    size_t biasCount = biases.size();
    outFile.write(reinterpret_cast<const char*>(&biasCount), sizeof(biasCount));
    for (const auto& [id, bias] : biases) {
        outFile.write(reinterpret_cast<const char*>(&id), sizeof(id));
        outFile.write(reinterpret_cast<const char*>(&bias), sizeof(bias));
    }

    outFile.close();
    std::cout << "Weights and biases saved to " << filename << std::endl;
}

void NeuralNetwork::loadWeightsAndBiases(const string& filename) {
    // Open the file for binary input
    std::ifstream inFile(filename, std::ios::binary);
    if (!inFile) {
        std::cerr << "Error: Could not open file for loading weights and biases: " << filename << std::endl;
        return;
    }

    // Load weights
    unordered_map<int, double> newWeights;
    size_t weightCount;
    inFile.read(reinterpret_cast<char*>(&weightCount), sizeof(weightCount));
    for (size_t i = 0; i < weightCount; ++i) {
        int id;
        double weight;
        inFile.read(reinterpret_cast<char*>(&id), sizeof(id));
        inFile.read(reinterpret_cast<char*>(&weight), sizeof(weight));
        newWeights[id] = weight;
    }

    // Load biases
    unordered_map<int, double> newBiases;
    size_t biasCount;
    inFile.read(reinterpret_cast<char*>(&biasCount), sizeof(biasCount));
    for (size_t i = 0; i < biasCount; ++i) {
        int id;
        double bias;
        inFile.read(reinterpret_cast<char*>(&id), sizeof(id));
        inFile.read(reinterpret_cast<char*>(&bias), sizeof(bias));
        newBiases[id] = bias;
    }

    inFile.close();

    // Update weights and biases in the neural network
    edit_weights(newWeights);
    edit_biases(newBiases);

    std::cout << "Weights and biases loaded from " << filename << std::endl;
}

double NeuralNetwork::LearingRateDeacy(double current_learning_rate) const {
    if (totalRuns + runs <= 0) return 0;

    const double min_learning_rate = 1e-6;
    double ratio = static_cast<double>(runs) / static_cast<double>(totalRuns + runs);
    ratio = std::max(0.0, std::min(1.0, ratio));  // Clamp between 0 and 1

    // Exponential decay
    double decay_factor = ratio * ratio;
    double decay_amount = current_learning_rate * decay_factor;

    return std::min(decay_amount, current_learning_rate - min_learning_rate);
}

void NeuralNetwork::testNetwork(vector<double> inputs, vector<double> correct_outputs) {
    // Forward pass only - don't modify network
    int inputIndex = 0;
    for (auto& node : allNodes) {
        if (node.layer == 0) {
            node.setActivationValue(inputs[inputIndex]);
            inputIndex++;
        }
    }

    // Track non-zero inputs
    int nonzero_inputs = 0;
    double input_sum = 0;
    for (auto& node : allNodes) {
        if (node.layer == 0) {
            if (node.activation_value > 0) {
                nonzero_inputs++;
                input_sum += node.activation_value;
            }
        }
    }

    // Forward propagation
    int current_layer = 1;
    while (current_layer <= last_layer) {
        for (auto& node : allNodes) {
            if (node.layer == current_layer) {
                double connection_total = 0;
                for (const auto& pair : node.backward_connections) {
                    double contribution = pair.second->start_address->activation_value *
                                        pair.second->weight;
                    connection_total += contribution;
                }
                if (node.layer == last_layer) {
                    node.activation_value = sigmoid(connection_total - node.bias);
                } else {
                    node.activation_value = leaky_relu(connection_total - node.bias);
                }
            }
        }
        current_layer++;
    }

    // Check accuracy
    bool precise = true;
    bool vague = true;
    int output_idx = 0;

    // Find highest activation and its index
    double highest_activation = -1.0;
    int highest_idx = -1;
    int correct_idx = -1;

    // First pass - find target index and highest activation
    for (const auto& node : allNodes) {
        if (node.layer == last_layer) {
            if (correct_outputs[output_idx] == 1.0) {
                correct_idx = output_idx;
            }
            if (node.activation_value > highest_activation) {
                highest_activation = node.activation_value;
                highest_idx = output_idx;
            }
            output_idx++;
        }
    }

    // Check if highest activation matches correct index
    if (highest_idx == correct_idx) {
        guess_correct_count++;
    }

    // Reset output_idx for original precision checks
    output_idx = 0;

    // Original precision checks
    for (const auto& node : allNodes) {
        if (node.layer == last_layer) {
            double target = correct_outputs[output_idx++];
            double actual = node.activation_value;

            if (target == 1.0) {
                if (actual <= 0.9) vague = false;
                if (actual < 0.9) precise = false;
            } else {
                if (actual > 0.5) vague = false;
                if (actual > 0.1) precise = false;
            }
        }
    }

    if (precise) precise_correct_count++;
    if (vague) vauge_correct_count++;
}

void NeuralNetwork::clearConnections() {
    for (auto& node : allNodes) {
        node.forward_connections.clear();
        node.backward_connections.clear();
    }
    allConnections.clear();
}

void NeuralNetwork::clearNodes() {
    allNodes.clear();
}

ThreadPool::ThreadPool(size_t numThreads) : stop(false) {
    for(size_t i = 0; i < numThreads; ++i)
        workers.emplace_back([this] {
            while(true) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lock(queueMutex);
                    condition.wait(lock, [this] {
                        return stop || !tasks.empty();
                    });
                    if(stop && tasks.empty())
                        return;
                    task = std::move(tasks.front());
                    tasks.pop();
                }
                task();
            }
        });
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock<std::mutex> lock(queueMutex);
        stop = true;
    }
    condition.notify_all();
    for(auto& worker : workers)
        if(worker.joinable())
            worker.join();
}

ThreadNetworks::ThreadNetworks(int number_networks, double lower_learning_rate,
                             double upper_learning_rate, std::vector<double>& startingWeights,
                             std::vector<double>& startingBiases, int input_node_count,
                             int hidden_layer_count, int node_per_hidden_layer,
                             int output_node_count, int backprop_after)
    : threadPool(std::thread::hardware_concurrency() - 1),
      batchSize(calculateOptimalBatchSize(number_networks)), upperBackProp(backprop_after) {

    networks_.reserve(number_networks);
    processingComplete.resize(number_networks, false);

    int numImages = 1000;
    int numRows = 28;
    int numCols = 28;

    double learning_rate_step = (upper_learning_rate - lower_learning_rate) / (number_networks - 1);
    testImages = readMNISTImages(TestDataFileImage, numImages, numRows, numCols);
    testLabels = readMNISTLabels(TestDataFileLabel, numImages);

    for (int i = 0; i < number_networks; i++) {
        double current_learning_rate = lower_learning_rate + (i * learning_rate_step);
        networks_.push_back(std::make_unique<NeuralNetwork>(
            input_node_count, hidden_layer_count, node_per_hidden_layer,
            output_node_count, current_learning_rate, startingWeights, startingBiases,backprop_after));
    }
}

ThreadNetworks::ThreadNetworks(int number_networks, double lower_learning_rate,
                             double upper_learning_rate, int input_node_count,
                             int hidden_layer_count_, int node_per_hidden_layer,
                             int output_node_count, const string& FilePath, int backprop_after)
    : threadPool(std::thread::hardware_concurrency() - 1),
      batchSize(calculateOptimalBatchSize(number_networks)), upperBackProp(backprop_after){

    networks_.reserve(number_networks);
    processingComplete.resize(number_networks, false);

    int numImages = 1000;
    int numRows = 28;
    int numCols = 28;

    double learning_rate_step = (upper_learning_rate - lower_learning_rate) / (number_networks - 1);
    testImages = readMNISTImages(TestDataFileImage, numImages, numRows, numCols);
    testLabels = readMNISTLabels(TestDataFileLabel, numImages);

    for (int i = 0; i < number_networks; i++) {
        double current_learning_rate = lower_learning_rate + (i * learning_rate_step);
        networks_.push_back(make_unique<NeuralNetwork>(input_node_count, hidden_layer_count_,node_per_hidden_layer,
                  output_node_count, current_learning_rate, FilePath, backprop_after));
    }
}

ThreadNetworks::ThreadNetworks(int number_networks, double lower_learning_rate,
                             double upper_learning_rate, int input_node_count,
                             int hidden_layer_count_, int node_per_hidden_layer,
                             int output_node_count, const string& FilePath, int backprop_after, bool fileSorting)
    : threadPool(std::thread::hardware_concurrency() - 1),
      batchSize(calculateOptimalBatchSize(number_networks)), upperBackProp(backprop_after)  {

    networks_.reserve(number_networks);
    processingComplete.resize(number_networks, false);

    int numImages = 1000;
    int numRows = 28;
    int numCols = 28;

    double learning_rate_step = (upper_learning_rate - lower_learning_rate) / (number_networks - 1);
    testImages = readMNISTImages(TestDataFileImage, numImages, numRows, numCols);
    testLabels = readMNISTLabels(TestDataFileLabel, numImages);

    for (int i = 0; i < number_networks; i++) {
        double current_learning_rate = lower_learning_rate + (i * learning_rate_step);
        networks_.push_back(make_unique<NeuralNetwork>(input_node_count, hidden_layer_count_,node_per_hidden_layer,
                  output_node_count, current_learning_rate, FilePath, backprop_after, fileSorting));
    }
}

ThreadNetworks::ThreadNetworks(int number_networks, double lower_learning_rate,
                             double upper_learning_rate, int input_node_count,
                             int hidden_layer_count_, int node_per_hidden_layer,
                             int output_node_count, queue<string>& FilePaths, bool fileSorting)
: threadPool(std::thread::hardware_concurrency() - 1),
      batchSize(calculateOptimalBatchSize(number_networks)){
    networks_.reserve(number_networks);
    processingComplete.resize(number_networks, false);


    upperBackProp = 105;
    int numImages = 1000;
    int numRows = 28;
    int numCols = 28;

    double learning_rate_step = (upper_learning_rate - lower_learning_rate) / (number_networks - 1);
    testImages = readMNISTImages(TestDataFileImage, numImages, numRows, numCols);
    testLabels = readMNISTLabels(TestDataFileLabel, numImages);

    for (int i = 0; i < number_networks; i++) {
        double current_learning_rate = lower_learning_rate + (i * learning_rate_step);
        string file = FilePaths.front();
        FilePaths.pop();
        networks_.push_back(make_unique<NeuralNetwork>(input_node_count,
                  hidden_layer_count_,node_per_hidden_layer,
                  output_node_count, current_learning_rate, file, fileSorting));
    }

}

ThreadNetworks::ThreadNetworks(int number_networks, double learning_rate,
                             int input_node_count, int base_hidden_layer_count,
                             int base_total_nodes, int output_node_count,
                             int backprop_after)
    : threadPool(std::thread::hardware_concurrency() - 1),
      batchSize(calculateOptimalBatchSize(number_networks)),
      upperBackProp(backprop_after)
{
    networks_.reserve(number_networks);
    processingComplete.resize(number_networks, false);

    // Setup test data
    int numImages = 1000;
    int numRows = 28;
    int numCols = 28;
    testImages = readMNISTImages(TestDataFileImage, numImages, numRows, numCols);
    testLabels = readMNISTLabels(TestDataFileLabel, numImages);

    // Create variations for each network
    for (int i = 0; i < number_networks; i++) {
        // Vary number of hidden layers (base ± 2)
        int hidden_layers = base_hidden_layer_count;
        if (i % 5 == 1) hidden_layers = max(1, base_hidden_layer_count - 2);
        if (i % 5 == 2) hidden_layers = max(1, base_hidden_layer_count - 1);
        if (i % 5 == 3) hidden_layers = base_hidden_layer_count + 1;
        if (i % 5 == 4) hidden_layers = base_hidden_layer_count + 2;

        vector<int> nodes_per_layer(hidden_layers);
        double total_nodes_used = 0;

        // Distribute nodes based on architecture pattern while maintaining total
        if (i % 4 == 0) {
            // Pyramidal (decreasing)
            double factor = base_total_nodes /
                          (hidden_layers * (1.0 + hidden_layers) / 2.0); // Sum of sequence 1 to n
            for (int l = 0; l < hidden_layers; l++) {
                nodes_per_layer[l] = max(32, static_cast<int>(factor * (hidden_layers - l)));
                total_nodes_used += nodes_per_layer[l];
            }
        }
        else if (i % 4 == 1) {
            // Inverse pyramidal (increasing)
            double factor = base_total_nodes /
                          (hidden_layers * (1.0 + hidden_layers) / 2.0);
            for (int l = 0; l < hidden_layers; l++) {
                nodes_per_layer[l] = max(32, static_cast<int>(factor * (l + 1)));
                total_nodes_used += nodes_per_layer[l];
            }
        }
        else if (i % 4 == 2) {
            // Diamond (peak in middle)
            double mid_point = (hidden_layers - 1) / 2.0;
            double max_height = base_total_nodes / hidden_layers;
            for (int l = 0; l < hidden_layers; l++) {
                double distance = abs(l - mid_point);
                nodes_per_layer[l] = max(32, static_cast<int>(
                    max_height * (1.0 - 0.5 * (distance / mid_point))));
                total_nodes_used += nodes_per_layer[l];
            }
        }
        else {
            // Uniform distribution
            int nodes_per_layer_uniform = base_total_nodes / hidden_layers;
            for (int l = 0; l < hidden_layers; l++) {
                nodes_per_layer[l] = max(32, nodes_per_layer_uniform);
                total_nodes_used += nodes_per_layer[l];
            }
        }

        // Generate weights and biases
        vector<double> startingWeights = generateStartingWeights(input_node_count, hidden_layers,
                                                               nodes_per_layer[0], output_node_count);
        vector<double> startingBiases = generateStartingBiases(hidden_layers,
                                                             accumulate(nodes_per_layer.begin(),
                                                                      nodes_per_layer.end(), 0) / hidden_layers,
                                                             output_node_count);

        // Create network with custom architecture
        networks_.push_back(make_unique<NeuralNetwork>(
            input_node_count, hidden_layers, nodes_per_layer[0],
            output_node_count, learning_rate, startingWeights, startingBiases,
            backprop_after
        ));

        // Log network architecture
        cout << "Network " << i << " architecture:" << endl;
        cout << "  Hidden layers: " << hidden_layers << endl;
        cout << "  Nodes per layer: ";
        for (int nodes : nodes_per_layer) {
            cout << nodes << " ";
        }
        cout << "\n  Total nodes: " << total_nodes_used << endl;
        cout << "  Average nodes per layer: " << total_nodes_used/hidden_layers << endl;
        cout << endl;
    }
}

void ThreadNetworks::runThreading(const std::vector<double>& image,
                                const std::vector<double>& correct_label_output) {
    if (shouldStop || costUpdateInProgress) return;

    {
        std::lock_guard<std::mutex> lock(resultsMutex);
        std::fill(processingComplete.begin(), processingComplete.end(), false);
        processedNetworks.store(0);
    }

    size_t total = networks_.size();
    const auto timeout = std::chrono::seconds(5);
    const auto startTime = std::chrono::steady_clock::now();

    std::vector<std::future<void>> allFutures;
    allFutures.reserve(total);

    // Submit all tasks at once with better error handling
    for (size_t i = 0; i < total && !shouldStop; ++i) {
        auto future = threadPool.enqueue([this, i, &image, &correct_label_output] {
            try {
                if (!shouldStop) {
                    if (i >= networks_.size()) {
                        throw std::runtime_error("Network index out of bounds");
                    }
                    trainNetwork(i, image, correct_label_output);
                    processedNetworks.fetch_add(1, std::memory_order_relaxed);
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(resultsMutex);
                std::cerr << "Critical error in network " << i << ": " << e.what() << std::endl;
                shouldStop = true;
            }
        });
        allFutures.push_back(std::move(future));
    }

    // Wait for completion with timeout
    for (auto& future : allFutures) {
        if (future.valid()) {
            auto status = future.wait_for(std::chrono::milliseconds(100));
            if (status == std::future_status::timeout) {
                if (std::chrono::steady_clock::now() - startTime > timeout) {
                    shouldStop = true;
                    break;
                }
            }
        }
    }
}

void ThreadNetworks::trainNetwork(size_t networkIndex, const vector<double>& input,
                                const vector<double>& correct_output) {
    lock_guard<mutex> lock(resultsMutex);

    auto& network = networks_[networkIndex];
    int backProp = network->backprop_count;
    int upperBack = network->upper_backprop_count;
    network->backpropigate_network(input, correct_output);

    if(backProp +1 == upperBack) {
        lock_guard<mutex> test_lock(test_data_mutex);

        // Run test batch
        for(int i = 0; i < upperBack; i++) {
            int j = getRandom(0, testImages.size() - 1);
            vector<double> test_label(10, 0.0);
            test_label[testLabels[j]] = 1.0;
            network->testNetwork(testImages[j], test_label);
        }
        PrintCost(networkIndex);
    }
}

void ThreadNetworks::PrintCost(int netwokIndex) {
    if (costUpdateInProgress.exchange(true)) return;
    lock_guard<mutex> lock(costMutex);
    try {
        if (!window_) {
            costUpdateInProgress = false;
            return;
        }

        auto& network = networks_[netwokIndex];
        double cost = network->getCost();
        int precise_count = network->precise_correct_count;
        int vague_count = network->vauge_correct_count;
        int guess_count = network->guess_correct_count;  // Get new counter

        window_->setLearningRate(network->ID, network->getLearningRate());
        window_->addDataPoint(network->ID, cost);

        cout << network->getLearningRate() << ": (GUESS) "
             << guess_count << "/" << upperBackProp << " (VAGUE) "
             << vague_count << "/" << upperBackProp << " (PRECISE) "
             << precise_count << "/" << upperBackProp
             << " Cost: " << cost << endl;
        cout << endl;

        network->resetStatistics();
    } catch (const exception& e) {
        cerr << "Error in PrintCost: " << e.what() << endl;
    }

    costUpdateInProgress = false;
}

size_t ThreadNetworks::calculateOptimalBatchSize(size_t numNetworks) {
    const size_t hardwareConcurrency = std::thread::hardware_concurrency();
    return std::max(size_t(1),
        std::min(numNetworks,
            ((numNetworks + hardwareConcurrency - 1) / hardwareConcurrency) * hardwareConcurrency));
}
