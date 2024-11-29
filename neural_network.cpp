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
    double connection_total = 0;
    if (layer != 0) {
        int active_inputs = 0;

        for (const auto& pair : backward_connections) {
            double input_val = pair.second->start_address->activation_value;
            double weight = pair.second->weight;
            double contribution = input_val * weight;
            connection_total += contribution;

            if (input_val != 0) {
                active_inputs++;
            }
        }
        double pre_activation = connection_total - bias;

        activation_value = relu(pre_activation);
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
                const int MIN_NODES = 64;  // Don't go below this number
                for (int l = 0; l < hLayer_count; l++) {
                    for (int n = 0; n < current_layer_nodes; n++) {
                        Node newHiddenNode(l+1, next_ID, _startingBiases[n*(l+1)]);
                        allNodes.push_back(newHiddenNode);
                    }
                    current_layer_nodes = max(MIN_NODES, current_layer_nodes / 2);  // Halve but don't go below minimum
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
    const int MIN_NODES = 64;  // Don't go below this number
    for (int l = 0; l < hLayer_count; l++) {
        for (int n = 0; n < current_layer_nodes; n++) {
            Node newHiddenNode(l+1, next_ID, 0);
            allNodes.push_back(newHiddenNode);
        }
        current_layer_nodes = max(MIN_NODES, current_layer_nodes / 2);  // Halve but don't go below minimum
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
    const int MIN_NODES = 64;  // Don't go below this number
    for (int l = 0; l < hLayer_count; l++) {
        for (int n = 0; n < current_layer_nodes; n++) {
            Node newHiddenNode(l+1, next_ID, 0);
            allNodes.push_back(newHiddenNode);
        }
        current_layer_nodes = max(MIN_NODES, current_layer_nodes / 2);  // Halve but don't go below minimum
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

void NeuralNetwork::run_network(vector<double> inputs, vector<double> correct_outputs) {
    // Set input values (keeping your existing code)
    int inputIndex = 0;
    for (auto& node : allNodes) {
        if (node.layer == 0) {
            node.setActivationValue(inputs[inputIndex]);
            inputIndex++;
        }
    }

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

   // Forward propagation through layers using ReLU for hidden layers
   int current_layer = 1;
   while (current_layer <= last_layer) {
       for (auto& node : allNodes) {
           if (node.layer == current_layer) {
               double connection_total = 0;
               for (const auto& pair : node.backward_connections) {
                   double contribution = pair.second->start_address->activation_value * pair.second->weight;
                   connection_total += contribution;
               }
               if (node.layer == last_layer) {
                   // Keep sigmoid for output layer
                   node.activation_value = sigmoid(connection_total - node.bias);
               } else {
                   // Use ReLU for hidden layers
                   node.activation_value = relu(connection_total - node.bias);
               }
           }
       }
       current_layer++;
   }

   // Calculate network outputs, cost, and errors for backpropagation
   int output_count = 0;
   double total_cost = 0.0;

   // Calculate error for output layer (using sigmoid derivative since output layer uses sigmoid)
   for(auto& node : allNodes) {
       if(node.layer == last_layer) {
           // Get target and actual values
           double target = correct_outputs[output_count];
           double actual = node.activation_value;
           // Calculate pure cost (MSE component)
           double error = target - actual;
           total_cost += pow(error, 2);

           // Calculate error value for output layer using sigmoid derivative
           node.error_value = actual * (1 - actual) * error;
           output_count++;
       }
   }

   // Calculate error for hidden layers using ReLU derivative

   for(int i = last_layer - 1; i > 0; i--) {

       for(auto& node : allNodes) {
           if(node.layer == i) {

               // Sum error values from next layer through connections
               double error_sum = 0.0;
               for(auto& connection : node.forward_connections) {
                   double contribution = connection.second->weight * connection.second->end_address->error_value;
                   error_sum += contribution;

               }


               // ReLU derivative is 1 if input was positive, 0 if input was negative
               node.error_value = relu_derivative(node.activation_value) * error_sum;
           }
       }
   }
   // Store the cost for monitoring (thread-safe)

   {
       std::lock_guard<std::mutex> lock(cost_mutex);
       average_cost.push_back(total_cost);
   }
}

void NeuralNetwork:: resetStatistics() {
    std::lock_guard<std::mutex> lock(statsMutex);
    average_cost.clear();
    precise_correct_count = 0;
    vauge_correct_count = 0;
}

void NeuralNetwork::backpropigate_network() {

    save_counter++;
    if (save_counter >= SAVE_INTERVAL) {
        saveNetworkData();
        save_counter = 0;
    }
    static const double initial_lr_scale = 0.1;  // Start with 10% of learning rate
    static const int warmup_steps = 1000;        // Gradually increase over 1000 steps

    // Calculate effective learning rate with warmup
    double current_lr = learning_rate;
    if (backprop_count < warmup_steps) {
        current_lr *= (initial_lr_scale + (1.0 - initial_lr_scale) *
                      static_cast<double>(backprop_count) / warmup_steps);
    }

    unordered_map<int, double> newWeights;
    unordered_map<int, double> newBiases;

    // Calculate new weights
    for (auto& connection : allConnections) {
        double nodeError = connection.second.end_address->error_value;
        double raw_weight_change = current_lr * nodeError *
                                 connection.second.start_address->activation_value;

        // Add L2 regularization
        raw_weight_change -= weight_decay * connection.second.weight;

        // Apply momentum
        double weightChange = raw_weight_change;
        if (prev_weight_changes.find(connection.second.ID) != prev_weight_changes.end()) {
            weightChange += momentum * prev_weight_changes[connection.second.ID];
        }

        prev_weight_changes[connection.second.ID] = weightChange;

        // Gradient clipping
        double new_weight = connection.second.weight + weightChange;
        new_weight = std::max(-0.5, std::min(0.5, new_weight));

        newWeights[connection.second.ID] = new_weight;
    }

    // Calculate new biases
    for (auto& node : allNodes) {
        if (node.layer != 0) {
            double biasChange = current_lr * node.error_value;

            // Apply momentum to bias changes
            if (prev_bias_changes.find(node.ID) != prev_bias_changes.end()) {
                biasChange += momentum * prev_bias_changes[node.ID];
            }

            prev_bias_changes[node.ID] = biasChange;

            // Clip bias
            double new_bias = node.bias + biasChange;
            new_bias = std::max(-0.5, std::min(0.5, new_bias));

            newBiases[node.ID] = new_bias;
        }
    }

    weights_.push_back(newWeights);
    biases_.push_back(newBiases);

    backprop_count++;
    if(backprop_count%(upper_backprop_count/10) == 0) {
        cout << "Backprop " << backprop_count << "/" <<upper_backprop_count << endl;
    }

    if(backprop_count == upper_backprop_count) {
        auto avgWeights = average(weights_);
        auto avgBiases = average(biases_);

        edit_weights(avgWeights);
        edit_biases(avgBiases);

        weights_.clear();
        biases_.clear();

        // Gradual learning rate decay
        learning_rate *= 0.9999;
        learning_rate = std::max(0.00001, learning_rate);

        backprop_count = 0;
    }
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

double NeuralNetwork:: getCost() {
    std::lock_guard<std::mutex> lock(cost_mutex);
    if (average_cost.empty()) {
        return current_cost;
    }

    double total_cost = 0.0;
    int count = 0;

    for (double cost : average_cost) {
        if (std::isfinite(cost)) {
            total_cost += cost;
            count++;
        }
    }

    if (count == 0) {
        return current_cost;
    }

    current_cost = total_cost / count;
    cost_sample_count = count;

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
            node.setActivationValue(inputs[inputIndex++]);
        }
    }

    // Forward propagation
    for(int layer = 1; layer <= last_layer; layer++) {
        for (auto& node : allNodes) {
            if (node.layer == layer) {
                double sum = 0.0;
                for (const auto& conn : node.backward_connections) {
                    sum += conn.second->start_address->activation_value *
                          conn.second->weight;
                }
                node.activation_value = (layer == last_layer) ?
                    sigmoid(sum - node.bias) : relu(sum - node.bias);
            }
        }
    }

    // Check accuracy
    bool precise = true;
    bool vague = true;
    int output_idx = 0;

    for (const auto& node : allNodes) {
        if(node.layer == last_layer) {
            double target = correct_outputs[output_idx++];
            double actual = node.activation_value;

            if(target == 1.0) {
                if(actual <= 0.9) vague = false;
                if(actual < 0.9) precise = false;
            } else {
                if(actual > 0.3) vague = false;
                if(actual > 0.1) precise = false;
            }
        }
    }

    if(precise) precise_correct_count++;
    if(vague) vauge_correct_count++;
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
    network->run_network(input, correct_output);
    int backProp = network->backprop_count;
    int upperBack = network->upper_backprop_count;
    network->backpropigate_network();

    if(backProp +1 == upperBack) {
        lock_guard<mutex> test_lock(test_data_mutex);

        // Run test batch
        for(int i = 0; i < 100; i++) {
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

            // Update window data
            window_->setLearningRate(network->ID, network->getLearningRate());
            window_->addDataPoint(network->ID, cost);

            // Print network stats
        cout << network->getLearningRate() << ": (VAGUE) "
                 << vague_count << "/" << upperBackProp << "(PRECISE) "
                 << precise_count << "/" << upperBackProp
                 << "Cost: " << cost << endl;
        cout << endl; // Add blank line between batches
        // Reset statistics for next batch

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
