#pragma once

#include <iostream>
#include <map>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <utility>
#include <chrono>
#include <thread>
#include <SFML/Graphics.hpp>
#include "neural_network.h"

using namespace std;

class Visualizer {
    public:
        Visualizer(unsigned int width, unsigned int height, const std::string& title, NeuralNetwork &NETWORK)
                : window(sf::VideoMode(width, height), title) {
            window.setFramerateLimit(60);  // Set frame rate limit
            network = &NETWORK;
            window_width = width;
            window_height = height;
        }

        void run() {
            while (window.isOpen()) {
                vector<float> inputs;
                for(int i = 0; i < 3; i++) {
                    inputs.push_back((static_cast<double>(rand()) / RAND_MAX) * 2.0 - 1.0);
                }
                network->run_network(inputs);
                handle_events();
                update();
                render();
                this_thread::sleep_for(std::chrono::seconds(1));
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

        void update() {

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

        for (int layer = 0; layer < total_layers; ++layer) {
            int nodes_in_layer = network->nodes_in_layer(layer);
            float x_pos = x_padding + layer * x_layer_spacing;
            float total_node_height = (nodes_in_layer - 1.0f) * (window_height / (nodes_in_layer + 1.0f));
            float y_start = (window_height / 2) - (total_node_height / 2);

            int node_count = 0;
            for (const auto& node : network->allNodes) {
                if (node.layer == layer) {
                    float y_pos = y_start + node_count * (window_height / (nodes_in_layer + 1));

                    sf::CircleShape circle(circle_radius);
                    circle.setPosition(x_pos - circle_radius, y_pos - circle_radius);

                    sf::Uint8 alpha = static_cast<sf::Uint8>(std::abs(node.activation_value) * 255);
                    if (layer == Node::last_layer) {
                        circle.setFillColor(node.activation_value < 0 ? sf::Color(255, 0, 0, alpha) : sf::Color(0, 255, 0, alpha));
                    } else {
                        circle.setFillColor(sf::Color(255, 255, 255, alpha));
                    }

                    circle.setOutlineColor(sf::Color::White);
                    circle.setOutlineThickness(2.f);

                    window.draw(circle);
                    ++node_count;
                }
            }
        }
        window.display();
    }
};