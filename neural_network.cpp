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

NeuralNetwork::NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count, double _learning_rate)
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

NeuralNetwork::NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count, double _learning_rate,
                unordered_map<int, double>_startingWeights, unordered_map<int, double> _startingBiases)
                    : learning_rate(_learning_rate), next_ID(0), connection_ID(0), last_layer(0), ID(nextID++) {

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


NeuralNetwork::NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count,
    double _learning_rate,const string& FilePath)
    :learning_rate(_learning_rate), next_ID(0), connection_ID(0), last_layer(0), ID(nextID++)
{

    for (int n = 0; n < iNode_count; n++) {
        Node newInputNode(0, next_ID);
        allNodes.push_back(newInputNode);
    }
    last_layer++;

    // Create hidden layers
    for (int l = 0; l < hLayer_count; l++) {
        for (int n = 0; n < hNode_count; n++) {
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
    saveNetworkData();
    clearConnections();
    clearNodes();
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
                        runs++;
                        backprop_count = 0;
                        learning_rate -= LearingRateDeacy(learning_rate);
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

double NeuralNetwork:: LearingRateDeacy(double learning_rate) {

    if (totalRuns + runs <= 0) return learning_rate;

    const double min_learning_rate = 1e-6;
    double ratio = static_cast<double>(runs) / static_cast<double>(totalRuns + runs);
    ratio = std::max(0.0, std::min(1.0, ratio));  // Clamp between 0 and 1
    double decay_factor = ratio * ratio;

    double decayed_learning_rate = learning_rate * decay_factor;
    return max(min_learning_rate, decayed_learning_rate);
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
                             int output_node_count)
    : threadPool(std::thread::hardware_concurrency() - 1),
      batchSize(calculateOptimalBatchSize(number_networks)) {

    networks_.reserve(number_networks);
    networkOutputs.resize(number_networks, 0.0);
    networkErrors.resize(number_networks, 0.0);
    processingComplete.resize(number_networks, false);

    double learning_rate_step = (upper_learning_rate - lower_learning_rate) / (number_networks - 1);

    for (int i = 0; i < number_networks; i++) {
        double current_learning_rate = lower_learning_rate + (i * learning_rate_step);
        networks_.push_back(std::make_unique<NeuralNetwork>(
            input_node_count, hidden_layer_count, node_per_hidden_layer,
            output_node_count, current_learning_rate, startingWeights, startingBiases));
    }
}

ThreadNetworks::ThreadNetworks(int number_networks, double lower_learning_rate,
                             double upper_learning_rate, int input_node_count,
                             int hidden_layer_count_, int node_per_hidden_layer,
                             int output_node_count, const std::string& WeightFilePath,
                             const std::string& BaisFilePath)
    : threadPool(std::thread::hardware_concurrency() - 1),
      batchSize(calculateOptimalBatchSize(number_networks)) {

    auto weights = loadData(WeightFilePath);
    auto bias = loadData(BaisFilePath);

    networks_.reserve(number_networks);
    networkOutputs.resize(number_networks, 0.0);
    networkErrors.resize(number_networks, 0.0);
    processingComplete.resize(number_networks, false);

    double learning_rate_step = abs((upper_learning_rate - lower_learning_rate) / (number_networks - 1));

    for (int i = 0; i < number_networks; i++) {
        double current_learning_rate = lower_learning_rate + (i * learning_rate_step);
        networks_.push_back(std::make_unique<NeuralNetwork>(
            input_node_count, hidden_layer_count_, node_per_hidden_layer,
            output_node_count, current_learning_rate, weights, bias));
    }
}

void ThreadNetworks::runThreading(const std::vector<double>& image,
                                const std::vector<double>& correct_label_output) {
    if (shouldStop || costUpdateInProgress) return;

    std::fill(processingComplete.begin(), processingComplete.end(), false);
    processedNetworks.store(0);

    size_t total = networks_.size();
    const auto timeout = std::chrono::seconds(5);
    const auto startTime = std::chrono::steady_clock::now();

    // Process networks in batches
    for (size_t i = 0; i < total && !shouldStop; i += batchSize) {
        size_t batchEnd = std::min(i + batchSize, total);
        std::vector<std::future<void>> batchFutures;

        // Submit batch tasks
        for (size_t j = i; j < batchEnd && !shouldStop; ++j) {
            auto future = threadPool.enqueue([this, j, &image, &correct_label_output] {
                if (!shouldStop) {
                    trainNetwork(j, image, correct_label_output);
                    processedNetworks.fetch_add(1);
                }
            });
            batchFutures.push_back(std::move(future));
        }

        // Wait for batch completion with timeout
        for (auto& future : batchFutures) {
            if (future.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout) {
                if (std::chrono::steady_clock::now() - startTime > timeout) {
                    shouldStop = true;
                    break;
                }
            }
        }
    }

    if (!shouldStop) {
        synchronizeResults();
    }
}

void ThreadNetworks::trainNetwork(size_t networkIndex, const std::vector<double>& input,
                                const std::vector<double>& correct_output) {
    try {
        if (networkIndex >= networks_.size()) return;

        auto& network = networks_[networkIndex];
        double output = network->run_network(input, correct_output);

        {
            std::lock_guard<std::mutex> lock(resultsMutex);
            networkOutputs[networkIndex] = output;
            processingComplete[networkIndex] = true;
        }

        network->backpropigate_network();

    } catch (const std::exception& e) {
        std::cerr << "Error in network " << networkIndex << ": " << e.what() << std::endl;
    }
}

void ThreadNetworks::synchronizeResults() {
    std::lock_guard<std::mutex> lock(resultsMutex);

    for (size_t i = 0; i < networks_.size(); ++i) {
        if (processingComplete[i]) {
            networks_[i]->updateStatistics(networkOutputs[i], networkErrors[i]);
        }
    }
}

void ThreadNetworks::PrintCost() {
    if (costUpdateInProgress.exchange(true)) return;  // Prevent concurrent cost updates

    std::lock_guard<std::mutex> lock(costMutex);

    try {
        if (!window_) {
            costUpdateInProgress = false;
            return;
        }

        for (auto& network : networks_) {
            if (shouldStop) break;

            window_->setLearningRate(network->ID, network->getLearningRate());

            double cost = network->getCost();
            if (std::isfinite(cost)) {  // Check for valid cost value
                window_->addDataPoint(network->ID, cost);

                std::cout << network->getLearningRate() << ": (VAGUE) "
                         << network->vauge_correct_count << "/100 (PRECISE) "
                         << network->precise_correct_count << "/100" << std::endl;

                network->vauge_correct_count = 0;
                network->precise_correct_count = 0;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error in PrintCost: " << e.what() << std::endl;
    }

    costUpdateInProgress = false;
}

size_t ThreadNetworks::calculateOptimalBatchSize(size_t numNetworks) {
    const size_t hardwareConcurrency = std::thread::hardware_concurrency();
    return std::max(size_t(1),
        std::min(numNetworks,
            ((numNetworks + hardwareConcurrency - 1) / hardwareConcurrency) * hardwareConcurrency));
}
