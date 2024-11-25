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
            startingWeights.push_back(getRandom(-1, 1));
        }

        // Weights for connections between hidden layers
        for (int i = 0; i < (number_hidden_layers - 1) * number_node_per_hidden * number_node_per_hidden; i++) {
            startingWeights.push_back(getRandom(-1, 1));
        }

        // Weights for connections from last hidden layer to output layer
        for (int i = 0; i < number_node_per_hidden * output_layer; i++) {
            startingWeights.push_back(getRandom(-1, 1));
        }

        return startingWeights;
    }

    inline vector<double> generateStartingBiases(int number_hidden_layers, int number_node_per_hidden, int output_layer) {
         vector<double> startingBiases;

        // Biases for hidden layers
        for (int i = 0; i < number_hidden_layers * number_node_per_hidden; i++) {
            startingBiases.push_back(getRandom(-15.0,15.0));
        }

        // Biases for output layer
        for (int i = 0; i < output_layer; i++) {
            startingBiases.push_back(getRandom(-15.0,15.0));
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
        int total_outputs = 0;

        //NeuralNetwork(){}
        NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count, double _learning_rate, int backprop_after);

        NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count, double _learning_rate,
            vector<double>_startingWeights, vector<double> _startingBiases, int backprop_after );

        NeuralNetwork(int iNode_count, int hLayer_count, int hNode_count, int oNode_count,
    double _learning_rate,const string& FilePath, int backprop_after);

        ~NeuralNetwork();
        double run_network(vector<double> inputs, vector<double> correct_outputs);

        void backpropigate_network();

        pair< unordered_map<int, double>, unordered_map<int, double>> getWeightsAndBiases();

        double getCost();

        double getLearningRate();

        void saveWeightsAndBiases(const string& filename);
        void loadWeightsAndBiases(const string& filename);

        void updateStatistics(double output, double error) {
            std::lock_guard<std::mutex> lock(statsMutex);
            // Update network statistics
            average_cost.push_back(output);
        }
    private:

        vector<Node> allNodes;
        unordered_map<int, connection> allConnections;
        vector<double> average_cost;
        vector<unordered_map<int, double>> weights_;
        vector<unordered_map<int, double>> biases_;
            std::mutex statsMutex;

        int totalRuns = 100;

        friend class ThreadNetworks;

        int runs = 0;
        int correct = 0;
        double learning_rate;

        int backprop_count = 0;
        int upper_backprop_count;

        // Instance-specific ID counters
        int next_ID;  // For nodes
        int connection_ID;  // For connections
        int last_layer;

        void edit_weights(const unordered_map<int, double>& new_values);

        void edit_biases(const unordered_map<int, double>& new_biases);

        void saveNetworkData();

        double LearingRateDeacy(double learning_rate);

        void clearConnections();
        void clearNodes();
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

        void SetWindow(GraphWindow& window) { window_ = &window; }
        void runThreading(const std::vector<double>& image, const std::vector<double>& correct_label_output);
        void PrintCost();
        void render() { if (window_) window_->render(); }
        bool isRunning() const { return !shouldStop; }
        void stop() { shouldStop = true; }

    private:
        int NetworkID = 0;
        std::vector<std::unique_ptr<NeuralNetwork>> networks_;
        GraphWindow* window_{nullptr};
        ThreadPool threadPool;
        std::mutex resultsMutex;
        std::mutex costMutex;
        const size_t batchSize;
        std::atomic<bool> shouldStop{false};
        std::atomic<bool> costUpdateInProgress{false};

        std::vector<double> networkOutputs;
        std::vector<double> networkErrors;
        std::vector<bool> processingComplete;
        std::atomic<size_t> processedNetworks{0};

        void trainNetwork(size_t networkIndex, const std::vector<double>& input,
                         const std::vector<double>& correct_output);
        void synchronizeResults();
        static size_t calculateOptimalBatchSize(size_t numNetworks);
    };

};

