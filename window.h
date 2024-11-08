#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <utility>
#include <chrono>
#include <fstream>
#include <thread>
#include <SFML/Graphics.hpp>
#include "neural_network.h"

using namespace std;

std::vector<std::vector<double>> readMNISTImages(const std::string& filePath, int numImages, int numRows, int numCols) {
    std::ifstream file(filePath, std::ios::binary);
    std::vector<std::vector<double>> images;

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
            std::vector<double> image;
            for (int j = 0; j < numRows * numCols; ++j) {
                unsigned char pixel = 0;
                file.read(reinterpret_cast<char*>(&pixel), 1);
                image.push_back(static_cast<double>(pixel) / 255.0); // Normalize to [0, 1]
            }
            images.push_back(image);
        }
        file.close();
    } else {
        std::cerr << "Failed to open the file: " << filePath << "\n";
    }

    return images;
}
std::vector<int> readMNISTLabels(const std::string& filePath, int numLabels) {
    std::ifstream file(filePath, std::ios::binary);
    std::vector<int> labels;

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
        std::cerr << "Failed to open the file: " << filePath << "\n";
    }

    return labels;
}


class Visualizer {
    public:
    // Paths to your MNIST files
        std::string imageFilePath = "C:\\Users\\nfree\\OneDrive\\Desktop\\NerualNetworkScott\\train-images.idx3-ubyte";
        std::string labelFilePath = "C:\\Users\\nfree\\OneDrive\\Desktop\\NerualNetworkScott\\train-labels.idx1-ubyte";

        // Read images and labels
        int numImages = 100000; // Change this to read as many as you need
        int numRows = 28;
        int numCols = 28;

        std::vector<std::vector<double>> images = readMNISTImages(imageFilePath, numImages, numRows, numCols);
        std::vector<int> labels = readMNISTLabels(labelFilePath, numImages);

    int count = 0;
        vector<double> total_errors;
        vector<unordered_map<int, double>> all_weights;
        vector<unordered_map<int, double>> all_biases;

        Visualizer(unsigned int width, unsigned int height, const std::string& title, NeuralNetwork &NETWORK)
                : window(sf::VideoMode(width, height), title) {
            window.setFramerateLimit(60);  // Set frame rate limit
            network = &NETWORK;
            window_width = width;
            window_height = height;
        }

        void run() {
            while (window.isOpen()) {
                for (int i = 0; i < images.size(); ++i) {
                    handle_events();
                    update(i);
                    if(count == 1) {
                        render();
                    }
                    //this_thread::sleep_for(std::chrono::milliseconds(500));
                }
            }
        }

    private:
        sf::RenderWindow window;
        NeuralNetwork* network;

        float window_width;
        float window_height;

        void handle_events() {
            sf::Event event;
            while (window.pollEvent(event)) {
                if (event.type == sf::Event::Closed)
                    window.close();
            }
        }

        void update(int output_index) {
                vector<double> correct_label_output;
                correct_label_output.resize(10);
                for (int output = 0; output < 10; output++) {
                    if (labels[output_index] == output) {
                        correct_label_output[output] = 1.0;
                    } else {
                        correct_label_output[output] = 0.0;
                    }
                }
                total_errors.push_back(network->run_network(images[output_index], correct_label_output));
                pair<unordered_map<int, double>, unordered_map<int, double>> network_output = network->backpropigate_network(); // Use the label as the correct node
                all_weights.emplace_back(network_output.first);
                all_biases.emplace_back(network_output.second);
                count++;

                // Average weights if necessary
                if(count % 100 == 0) {
                    cout << "RUN (" << count << "/" << "500) - " << endl;
                }
                if(count == 500)
                {
                    unordered_map<int, double> averaged_weights = network->average(all_weights);
                    unordered_map<int, double> averaged_biases = network->average(all_biases);
                    network->edit_weights(averaged_weights);
                    network->edit_biases(averaged_biases);
                    all_weights.clear();
                    all_biases.clear();
                    averaged_biases.clear();
                    averaged_weights.clear();
                    cout << "GENERATION COMPLETE - " << network->get_cost() << endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
                    total_errors.clear();
                    count = 0;
                }
        }

        void render() {
            window.clear(sf::Color::Black);

            int total_layers = Node::last_layer + 1;
            float x_padding = window_width * 0.1f;
            float x_layer_spacing = (window_width - 2 * x_padding) / (total_layers - 1);

            int max_nodes_in_layer = 0;
            for (int layer = 0; layer < total_layers; ++layer) {
                max_nodes_in_layer = std::max(max_nodes_in_layer, network->nodes_in_layer(layer));
            }

            float min_circle_radius = std::min(window_width / (2 * total_layers), window_height / (2 * max_nodes_in_layer + 1));
            float circle_radius = std::min(min_circle_radius, 10.0f);

            for(int layer = 0; layer < total_layers; layer++) {
                //get nodes of layer
                vector<Node> all_layer_nodes = network->get_nodes_by_layer(layer);

                //get only the first 20 as to not softlock program
                int numElements = std::min(20, static_cast<int>(all_layer_nodes.size()));
                vector<Node> nodes(all_layer_nodes.begin(), all_layer_nodes.begin() + numElements);
                int nodes_in_layer = nodes.size();

                float x_pos = x_padding + layer * x_layer_spacing;
                float total_node_height = (nodes_in_layer - 1.0f) * (window_height / (nodes_in_layer + 1.0f));
                float y_start = (window_height / 2) - (total_node_height / 2);

                int node_count = 0;

                for (const auto& node : nodes) {
                    if (node.layer == layer) {
                        float y_pos = y_start + node_count * (window_height / (nodes_in_layer + 1));

                        sf::CircleShape circle(circle_radius);
                        circle.setPosition(x_pos - circle_radius, y_pos - circle_radius);

                        sf::Uint8 alpha = static_cast<sf::Uint8>(std::abs(node.activation_value) * 255);
                        if (layer == Node::last_layer) {
                            circle.setFillColor(
                                    node.activation_value < 0 ? sf::Color(255, 0, 0, alpha) : sf::Color(0, 255, 0, alpha));
                        } else {
                            circle.setFillColor(sf::Color(255, 255, 255, alpha));
                        }

                        circle.setOutlineColor(sf::Color::White);
                        circle.setOutlineThickness(2.f);

                        window.draw(circle);
                        ++node_count;
                        for (const auto &connection: node.forward_connections) {
                            Node *start_node = connection.second->start_address;
                            Node *end_node = connection.second->end_address;
                            // Calculate start and end positions
                            sf::Vector2f start_pos(
                                    x_padding + start_node->layer * x_layer_spacing,
                                    (window_height / 2) - ((nodes_in_layer - 1)
                                    * (window_height / (nodes_in_layer + 1)) / 2)
                                    + (start_node->ID % nodes_in_layer)
                                    * (window_height / nodes_in_layer + 1)
                            );

                            sf::Vector2f end_pos(
                                    x_padding + end_node->layer * x_layer_spacing,
                                    (window_height / 2) - ((network->nodes_in_layer(connection.second->end_address->layer) - 1)
                                    * (window_height / (network->nodes_in_layer(connection.second->end_address->layer) + 1)) / 2)
                                    + (end_node->ID % network->nodes_in_layer(connection.second->end_address->layer))
                                    * (window_height / (network->nodes_in_layer(connection.second->end_address->layer) + 1))
                            );

                            //make line color green if positive and red if negative
                            sf::Color line_color =
                                    connection.second->weight > 0 ? sf::Color(0, 255, 0) : sf::Color(255, 0, 0);
                            float weight_magnitude = abs(connection.second->weight);
                            line_color.a = static_cast<sf::Uint8>(max(64.0f, weight_magnitude * 255.0f));

                            float line_thickness = 1.0f + weight_magnitude * 2.0f;

                            sf::VertexArray line(sf::Quads, 4);

                            sf::Vector2f direction = end_pos - start_pos;
                            sf::Vector2f unit_direction =
                                    direction / std::sqrt(direction.x * direction.x + direction.y * direction.y);
                            sf::Vector2f perpendicular(-unit_direction.y, unit_direction.x);

                            line[0].position = start_pos + perpendicular * (line_thickness / 2.0f);
                            line[1].position = start_pos - perpendicular * (line_thickness / 2.0f);
                            line[2].position = end_pos - perpendicular * (line_thickness / 2.0f);
                            line[3].position = end_pos + perpendicular * (line_thickness / 2.0f);

                            for (int i = 0; i < 4; ++i) {
                                line[i].color = line_color;
                            }

                            window.draw(line);
                        }
                    }
                }
            }
            window.display();
        }
};