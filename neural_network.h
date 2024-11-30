#pragma once
#include <iostream>
#include <map>
#include <vector>
#include <numeric>
#include <thread>
#include <mutex>
#include <valarray>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <random>
#include <unordered_map>
#include <queue>
#include <vector>
#include <map>
#include <cmath>
#include <future>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <string>
#include <filesystem>
#include <vector>
#include <thread>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <functional>
#include <memory>

#include "SFML/Graphics/Font.hpp"
#include "SFML/Graphics/RenderWindow.hpp"
#include "SFML/Window/Event.hpp"
#include "SFML/Graphics/RectangleShape.hpp"
#include "SFML/Graphics/Text.hpp"
#include "SFML/Graphics/CircleShape.hpp"

using namespace std;
using namespace filesystem;

namespace Rebecca {

    class ThreadNetworks;
    class Node;
    class NeuralNetwork;

    inline string DIR = "../network/";

    inline double getRandom(double min, double max) {
        static random_device rd;                            // Seed source
        static mt19937 gen(rd());                      // Mersenne Twister generator initialized with random seed
        uniform_real_distribution<> dis(min, max);   // Distribution in the range [min, max]
        return dis(gen);                                     // Return Random
    }

    inline double sigmoid(double x) {
        if (x > 88.0) return 1.0;
        if (x < -88.0) return 0.0;
        double result = 1.0 / (1.0 + std::exp(-x));
        return result;
    }


    inline void monitorNetworkSetup(const vector<double>& weights, const vector<double>& biases, int input_nodes, int hidden_layers, int nodes_per_hidden, int output_nodes) {
        cout << "\n=== Network Configuration ===" << endl;
        cout << "Input nodes: " << input_nodes << endl;
        cout << "Hidden layers: " << hidden_layers << endl;
        cout << "Initial nodes per hidden layer: " << nodes_per_hidden << endl;
        cout << "Output nodes: " << output_nodes << endl;

        cout << "\nWeight Statistics:" << endl;
        double weight_sum = 0;
        double weight_min = weights[0];
        double weight_max = weights[0];
        for(const auto& w : weights) {
            weight_sum += w;
            weight_min = min(weight_min, w);
            weight_max = max(weight_max, w);
        }
        cout << "Weight range: [" << weight_min << ", " << weight_max << "]" << endl;
        cout << "Weight average: " << weight_sum / weights.size() << endl;

        cout << "\nBias Statistics:" << endl;
        double bias_sum = 0;
        double bias_min = biases[0];
        double bias_max = biases[0];
        for(const auto& b : biases) {
            bias_sum += b;
            bias_min = min(bias_min, b);
            bias_max = max(bias_max, b);
        }
        cout << "Bias range: [" << bias_min << ", " << bias_max << "]" << endl;
        cout << "Bias average: " << bias_sum / biases.size() << endl;
    }

    inline double relu_derivative(double x) {
        return x > 0 ? 1.0 : 0.0;
    }

    double relu(double x) {
        return std::max(0.0, x);
    }

    double softmax(double x, double sum) {
        return std::exp(x) / sum;
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

    inline vector< vector<double>> readMNISTImages(const  string& filePath, int numImages, int numRows, int numCols) {
     ifstream file(filePath,  ios::binary);
     vector< vector<double>> images;

    if (file.is_open()) {
        int magicNumber = 0;
        int numberOfImages = 0;
        int rows = 0;
        int cols = 0;

        // Read and convert the magic number and header values
        file.read(reinterpret_cast<char*>(&magicNumber), 4);
        file.read(reinterpret_cast<char*>(&numberOfImages), 4);
        file.read(reinterpret_cast<char*>(&rows), 4);
        file.read(reinterpret_cast<char*>(&cols), 4);

        // Convert from big-endian to little-endian if needed
        magicNumber = __builtin_bswap32(magicNumber);
        numberOfImages = __builtin_bswap32(numberOfImages);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);

        for (int i = 0; i < numImages; ++i) {
             vector<double> image;
            for (int j = 0; j < numRows * numCols; ++j) {
                unsigned char pixel = 0;
                file.read(reinterpret_cast<char*>(&pixel), 1);
                image.push_back(static_cast<double>(pixel) / 255.0); // Normalize to [0, 1]
            }
            images.push_back(image);
        }
        file.close();
    } else {
         cerr << "Failed to open the file: " << filePath << "\n";
    }

    return images;
}

    inline vector<int> readMNISTLabels(const  string& filePath, int numLabels) {
         ifstream file(filePath,  ios::binary);
         vector<int> labels;

        if (file.is_open()) {
            int magicNumber = 0;
            int numberOfLabels = 0;

            // Read and convert the magic number and header values
            file.read(reinterpret_cast<char*>(&magicNumber), 4);
            file.read(reinterpret_cast<char*>(&numberOfLabels), 4);

            // Convert from big-endian to little-endian if needed
            magicNumber = __builtin_bswap32(magicNumber);
            numberOfLabels = __builtin_bswap32(numberOfLabels);

            for (int i = 0; i < numLabels; ++i) {
                unsigned char label = 0;
                file.read(reinterpret_cast<char*>(&label), 1);
                labels.push_back(static_cast<int>(label));
            }
            file.close();
        } else {
             cerr << "Failed to open the file: " << filePath << "\n";
        }

        return labels;
    }

    inline vector<double> generateStartingWeights(int input_layer, int number_hidden_layers, int number_node_per_hidden, int output_layer) {
        vector<double> startingWeights;

        // Weights for connections from input layer to first hidden layer
        for (int i = 0; i < input_layer * number_node_per_hidden; i++) {
            startingWeights.push_back(getRandom(-2, 2));
        }

        // Weights for connections between hidden layers
        for (int i = 0; i < (number_hidden_layers - 1) * number_node_per_hidden * number_node_per_hidden; i++) {
            startingWeights.push_back(getRandom(-2, 2));
        }

        // Weights for connections from last hidden layer to output layer
        for (int i = 0; i < number_node_per_hidden * output_layer; i++) {
            startingWeights.push_back(getRandom(-2, 2));
        }

        return startingWeights;
    }

    inline vector<double> generateStartingBiases(int number_hidden_layers, int number_node_per_hidden, int output_layer) {
        vector<double> startingBiases;

        // Biases for hidden layers
        for (int i = 0; i < number_hidden_layers * number_node_per_hidden; i++) {
            startingBiases.push_back(getRandom(0,0.2));
        }

        // Biases for output layer
        for (int i = 0; i < output_layer; i++) {
            startingBiases.push_back(getRandom(0,0.2));
        }

        return startingBiases;
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
        double batch_normalize(double x);

        void calculate_node();
    };

    class GraphWindow {
    public:
        bool run_network = true;

        GraphWindow(unsigned int width, unsigned int height, const std::string& title, ThreadNetworks* _allNetworks);
        ~GraphWindow();  // Added destructor

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
        int mouseX = -1;
        std::map<int, std::vector<double>> dataSets;
        std::map<int, double> learningRates;
        std::map<int, sf::Color> colors;
        std::mutex dataMutex;  // Added for thread safety
        bool isInitialized = false;  // Added for initialization checking

        void drawGraph();
        void drawCursorLineWithMarkers();
        void drawAxesLabels();
        void drawAxisTitles();
        void drawKey();
        sf::Color generateColor(int index);
        void drawYAxisLabel(double value, float x, float y);
        void drawXAxisLabel(int run, float x, float y);
        std::string formatLabel(double value);
        bool validateDataPoint(double value) const;
    };

    class ThreadPool {
    public:
        ThreadPool(size_t numThreads = std::max(1u, std::thread::hardware_concurrency() - 1));
        ~ThreadPool();

        template<class F, class... Args>
        auto enqueue(F&& f, Args&&... args)
            -> std::future<typename std::result_of<F(Args...)>::type> {
            using return_type = typename std::result_of<F(Args...)>::type;

            auto task = std::make_shared<std::packaged_task<return_type()>>(
                std::bind(std::forward<F>(f), std::forward<Args>(args)...));

            std::future<return_type> res = task->get_future();
            {
                std::unique_lock<std::mutex> lock(queueMutex);
                if(stop)
                    throw std::runtime_error("enqueue on stopped ThreadPool");

                tasks.emplace([task](){ (*task)(); });
            }
            condition.notify_one();
            return res;
        }

    private:
        std::vector<std::thread> workers;
        std::queue<std::function<void()>> tasks;
        std::mutex queueMutex;
        std::condition_variable condition;
        bool stop;
    };

    class NeuralNetwork {
    public:
        int ID;
        static int nextID;
        int vauge_correct_count = 0;
        int precise_correct_count = 0;

        using Matrix = std::valarray<std::valarray<double>>;

        using WeightMatrix = vector<vector<double>>;

        WeightMatrix convertToMatrix(const vector<unordered_map<int, double>>& maps) {
            size_t rows = maps.size();
            size_t maxCol = 0;

            // Find max column index
            for(const auto& map : maps) {
                for(const auto& [id, _] : map) {
                    maxCol = max(maxCol, static_cast<size_t>(id + 1));
                }
            }

            // Initialize matrix with zeros
            WeightMatrix matrix(rows, vector<double>(maxCol, 0.0));

            // Fill matrix
            for(size_t i = 0; i < maps.size(); i++) {
                for(const auto& [id, val] : maps[i]) {
                    matrix[i][id] = val;
                }
            }
            return matrix;
        }

        unordered_map<int, double> averageMatrix(const WeightMatrix& matrix) {
            unordered_map<int, double> result;
            if(matrix.empty()) return result;

            size_t cols = matrix[0].size();
            for(size_t j = 0; j < cols; j++) {
                double sum = 0.0;
                int count = 0;
                for(size_t i = 0; i < matrix.size(); i++) {
                    if(matrix[i][j] != 0.0) {
                        sum += matrix[i][j];
                        count++;
                    }
                }
                if(count > 0) {
                    result[j] = sum / count;
                }
            }
            return result;
        }

        unordered_map<int, double> batchAverage(const vector<unordered_map<int, double>>& updates) {
            auto matrix = convertToMatrix(updates);
            return averageMatrix(matrix);
        }

        //NeuralNetwork(){}
        NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count, double _learning_rate, int backprop_after);

        NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count, double _learning_rate,
            vector<double>_startingWeights, vector<double> _startingBiases, int backprop_after );

        NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count,
        double _learning_rate,const string& FilePath, int backprop_after);

        NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count,
        double _learning_rate,const string& FilePath, int backprop_after, bool fileSorting);

        ~NeuralNetwork();

        void backpropigate_network(const vector<double>& inputs, const vector<double>& correct_outputs);

        void resetStatistics();

        void testNetwork(vector<double> inputs, vector<double> correct_outputs);

        pair< unordered_map<int, double>, unordered_map<int, double>> getWeightsAndBiases();

        double getCost();

        double getLearningRate();

        void saveWeightsAndBiases(const string& filename);
        void loadWeightsAndBiases(const string& filename);
    private:

        mutable std::mutex cost_mutex;
        std::vector<double> average_cost;
        double current_cost = 0.0;
        int cost_sample_count = 0;

        vector<Node> allNodes;
        bool fileSorting = false;
        unordered_map<int, connection> allConnections;

        vector<unordered_map<int, double>> weights_;
        vector<unordered_map<int, double>> biases_;

        double momentum = 0.9;  // Momentum coefficient
        unordered_map<int, double> prev_weight_changes;
        unordered_map<int, double> prev_bias_changes;
        const double weight_decay = 0.001;

        std::mutex statsMutex;

        int totalRuns = 100;

        friend class ThreadNetworks;
        int runs = 0;
        int correct = 0;
        double learning_rate;

        int backprop_count = 0;
        int upper_backprop_count;

        const int SAVE_INTERVAL; // More frequent saving
        int save_counter = 0;


        // Instance-specific ID counters
        int next_ID;  // For nodes
        int connection_ID;  // For connections
        int last_layer;

        void edit_weights(const unordered_map<int, double>& new_values);

        void edit_biases(const unordered_map<int, double>& new_biases);

        void saveNetworkData();

        double LearingRateDeacy(double learning_rate) const;

        void clearConnections();
        void clearNodes();

        Node* find_node_by_id(int id) {
            for (auto& node : allNodes) {
                if (node.ID == id) return &node;
            }
            return nullptr;
        }
    };

    class ThreadNetworks {
    public:
        ThreadNetworks(int number_networks, double lower_learning_rate,
                      double upper_learning_rate, std::vector<double>& startingWeights,
                      std::vector<double>& startingBiases, int input_node_count,
                      int hidden_layer_count, int node_per_hidden_layer,
                      int output_node_count, int backprop_after);

        ThreadNetworks(int number_networks, double lower_learning_rate,
                             double upper_learning_rate, int input_node_count,
                             int hidden_layer_count_, int node_per_hidden_layer,
                             int output_node_count, const string& FilePath, int backprop_after);

        ThreadNetworks(int number_networks, double lower_learning_rate,
                             double upper_learning_rate, int input_node_count,
                             int hidden_layer_count_, int node_per_hidden_layer,
                             int output_node_count, const string& FilePath, int backprop_after, bool fliesorting);

        ThreadNetworks(int number_networks, double lower_learning_rate,
                             double upper_learning_rate, int input_node_count,
                             int hidden_layer_count_, int node_per_hidden_layer,
                             int output_node_count, queue<string>& FilePaths, bool fileSorting);

        void SetWindow(GraphWindow& window) { window_ = &window; }
        void runThreading(const std::vector<double>& image, const std::vector<double>& correct_label_output);
        void PrintCost(int netwokIndex);

        void render() { if (window_) window_->render(); }
        NeuralNetwork* getBestNetwork() const { return best_network.get(); }
        int getHighestPrecisionCount() const { return highest_precision_count; }
        bool isRunning() const { return !shouldStop; }
        void stop() { shouldStop = true; }

    private:
        int NetworkID = 0;

        std::vector<std::unique_ptr<NeuralNetwork>> networks_;
        GraphWindow* window_{nullptr};
        ThreadPool threadPool;
        int highest_precision_count = 0;
        unique_ptr<NeuralNetwork> best_network;
        int backpropCount = 0;
        int upperBackProp;
        int highest_local_precision_count = 0;
        std::mutex resultsMutex;
        std::mutex costMutex;
        std::mutex best_network_mutex;
        const size_t batchSize;

        const int VAGUE_THRESHOLD = 5;
        const int PRECISE_THRESHOLD = 3;
        bool thresholdMet = false;

        mutable mutex test_data_mutex;

        string TestDataFileImage = "Testing-Data-Images.idx3-ubyte";
        string TestDataFileLabel = "TestingData-labels.idx1-ubyte";

        vector<vector<double>> testImages;
        vector<int> testLabels;
        std::atomic<bool> shouldStop{false};
        std::atomic<bool> costUpdateInProgress{false};
        std::vector<bool> processingComplete;
        std::atomic<size_t> processedNetworks{0};

        int warmup_counter = 0;  // Track runs since last propagation
        const int WARMUP_PERIOD = 0;



        void trainNetwork(size_t networkIndex, const std::vector<double>& input,
                         const std::vector<double>& correct_output);

        static size_t calculateOptimalBatchSize(size_t numNetworks);
    };

};

