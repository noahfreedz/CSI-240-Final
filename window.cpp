#include "window.h"



using namespace std;

GraphWindow::GraphWindow(unsigned int width, unsigned int height, const std::string& title)
            : window(sf::VideoMode(width, height), title), window_width(width), window_height(height) {
        window.setFramerateLimit(60);
        if (!font.loadFromFile("Font.ttf")) {
            std::cerr << "Failed to load font\n";
        }
    }

bool GraphWindow:: isOpen() const {
        return window.isOpen();
    }

void GraphWindow:: handleEvents() {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed) {
                window.close();
            } else if (event.type == sf::Event::MouseMoved) {
                mouseX = event.mouseMove.x;
            }
        }
    }

void GraphWindow:: addDataPoint(int lineID, double value) {
        if (dataSets.find(lineID) == dataSets.end()) {
            dataSets[lineID] = std::vector<double>();
            colors[lineID] = generateColor(lineID);
        }
        dataSets[lineID].push_back(value);
    }

void GraphWindow:: setLearningRate(int lineID, double learningRate) {
        learningRates[lineID] = learningRate;
    }

void GraphWindow:: render() {
        window.clear(sf::Color::Black);
        drawGraph();
        drawAxesLabels();
        drawAxisTitles();
        drawKey();
        drawCursorLineWithMarkers();
        window.display();
    }

void GraphWindow:: drawGraph() {
        if (dataSets.empty()) return;

        double globalMax = -std::numeric_limits<double>::infinity();
        double globalMin = std::numeric_limits<double>::infinity();
        for (const auto& [id, data] : dataSets) {
            double maxVal = *std::max_element(data.begin(), data.end());
            double minVal = *std::min_element(data.begin(), data.end());
            if (maxVal > globalMax) globalMax = maxVal;
            if (minVal < globalMin) globalMin = minVal;
        }

        double midVal1 = globalMin + (globalMax - globalMin) * 0.25;
        double midVal2 = globalMin + (globalMax - globalMin) * 0.5;
        double midVal3 = globalMin + (globalMax - globalMin) * 0.75;

        sf::Vertex xAxis[] = {
                sf::Vertex(sf::Vector2f(100, window_height - 80)),
                sf::Vertex(sf::Vector2f(window_width - 80, window_height - 80))
        };
        sf::Vertex yAxis[] = {
                sf::Vertex(sf::Vector2f(100, 50)),
                sf::Vertex(sf::Vector2f(100, window_height - 80))
        };

        window.draw(xAxis, 2, sf::Lines);
        window.draw(yAxis, 2, sf::Lines);

        double yScale = (window_height - 160) / (globalMax - globalMin);

        for (const auto& [id, data] : dataSets) {
            double xSpacing = (window_width - 180) / static_cast<double>(data.size());
            std::vector<sf::Vertex> points;

            for (size_t i = 0; i < data.size(); ++i) {
                double x = 100 + i * xSpacing;
                double y = window_height - 80 - (data[i] - globalMin) * yScale;
                points.emplace_back(sf::Vertex(sf::Vector2f(x, y), colors[id]));
            }

            if (!points.empty()) {
                window.draw(&points[0], points.size(), sf::LinesStrip);
            }
        }
    }

void GraphWindow:: drawCursorLineWithMarkers() {
        if (mouseX < 100 || mouseX > window_width - 80) return;

        sf::Vertex cursorLine[] = {
                sf::Vertex(sf::Vector2f(mouseX, 50), sf::Color(200, 200, 200)),
                sf::Vertex(sf::Vector2f(mouseX, window_height - 80), sf::Color(200, 200, 200))
        };
        window.draw(cursorLine, 2, sf::Lines);

        double globalMax = -std::numeric_limits<double>::infinity();
        double globalMin = std::numeric_limits<double>::infinity();
        for (const auto& [id, data] : dataSets) {
            double maxVal = *std::max_element(data.begin(), data.end());
            double minVal = *std::min_element(data.begin(), data.end());
            if (maxVal > globalMax) globalMax = maxVal;
            if (minVal < globalMin) globalMin = minVal;
        }

        double yScale = (window_height - 160) / (globalMax - globalMin);
        double xSpacing = (window_width - 180) / static_cast<double>(dataSets.begin()->second.size());

        int index = static_cast<int>((mouseX - 100) / xSpacing);
        if (index < 0 || index >= dataSets.begin()->second.size()) return;

        for (const auto& [id, data] : dataSets) {
            if (index < data.size()) {
                double x = 100 + index * xSpacing;
                double y = window_height - 80 - (data[index] - globalMin) * yScale;

                // Draw a marker at the intersection
                sf::CircleShape marker(3);
                marker.setPosition(x - 3, y - 3);
                marker.setFillColor(colors[id]);
                window.draw(marker);

                // Draw a text label showing the run and cost
                sf::Text markerText;
                markerText.setFont(font);
                markerText.setString("Run: " + std::to_string(perRunCount * index) + "\nCost: " + formatLabel(data[index]));
                markerText.setCharacterSize(24);
                markerText.setFillColor(colors[id]);
                markerText.setPosition(x + 5, y - 15);
                window.draw(markerText);
            }
        }
    }

void GraphWindow:: drawAxesLabels() {
        if (dataSets.empty()) return;

        double globalMax = -std::numeric_limits<double>::infinity();
        double globalMin = std::numeric_limits<double>::infinity();
        for (const auto& [id, data] : dataSets) {
            double maxVal = *std::max_element(data.begin(), data.end());
            double minVal = *std::min_element(data.begin(), data.end());
            if (maxVal > globalMax) globalMax = maxVal;
            if (minVal < globalMin) globalMin = minVal;
        }

        double midVal1 = globalMin + (globalMax - globalMin) * 0.25;
        double midVal2 = globalMin + (globalMax - globalMin) * 0.5;
        double midVal3 = globalMin + (globalMax - globalMin) * 0.75;

        drawYAxisLabel(globalMin, 100, window_height - 80);
        drawYAxisLabel(midVal1, 100, window_height - 80 - (midVal1 - globalMin) * ((window_height - 160) / (globalMax - globalMin)));
        drawYAxisLabel(midVal2, 100, window_height - 80 - (midVal2 - globalMin) * ((window_height - 160) / (globalMax - globalMin)));
        drawYAxisLabel(midVal3, 100, window_height - 80 - (midVal3 - globalMin) * ((window_height - 160) / (globalMax - globalMin)));
        drawYAxisLabel(globalMax, 100, 50);

        size_t maxDataSize = 0;
        for (const auto& [id, data] : dataSets) {
            maxDataSize = std::max(maxDataSize, data.size());
        }

        size_t numLabels = std::min<size_t>(6, maxDataSize);
        if (numLabels > 1) {
            for (size_t i = 0; i < numLabels; ++i) {
                size_t index = i * (maxDataSize - 1) / (numLabels - 1);
                drawXAxisLabel(perRunCount * index, 100 + index * ((window_width - 180) / static_cast<double>(maxDataSize)), window_height - 70);
            }
        }
    }

void GraphWindow:: drawAxisTitles() {
        sf::Text yAxisTitle;
        yAxisTitle.setFont(font);
        yAxisTitle.setString("COST");
        yAxisTitle.setCharacterSize(20);
        yAxisTitle.setFillColor(sf::Color::White);
        yAxisTitle.setPosition(10, (window_height / 2) - (yAxisTitle.getLocalBounds().width / 2));
        yAxisTitle.setRotation(-90);
        window.draw(yAxisTitle);

        sf::Text xAxisTitle;
        xAxisTitle.setFont(font);
        xAxisTitle.setString("RUNS");
        xAxisTitle.setCharacterSize(20);
        xAxisTitle.setFillColor(sf::Color::White);
        xAxisTitle.setPosition((window_width / 2) - 30, window_height - 40);
        window.draw(xAxisTitle);
    }

void GraphWindow:: drawKey() {
        sf::Text keyTitle;
        keyTitle.setFont(font);
        keyTitle.setString("Learning Rates");
        keyTitle.setCharacterSize(16);
        keyTitle.setFillColor(sf::Color::White);
        keyTitle.setPosition(window_width - 180, 50);
        window.draw(keyTitle);

        int lineCount = 0;
        for (const auto& [id, rate] : learningRates) {
            sf::RectangleShape colorBox(sf::Vector2f(15, 15));
            colorBox.setFillColor(colors[id]);
            colorBox.setPosition(window_width - 180, 80 + lineCount * 30);
            window.draw(colorBox);

            sf::Text labelText;
            labelText.setFont(font);
            labelText.setString("LR = " + formatLabel(rate));
            labelText.setCharacterSize(14);
            labelText.setFillColor(sf::Color::White);
            labelText.setPosition(window_width - 155, 78 + lineCount * 30);
            window.draw(labelText);

            lineCount++;
        }
    }

sf::Color GraphWindow:: generateColor(int index) {
        static std::vector<sf::Color> colorPalette = {
                sf::Color::Red, sf::Color::Green, sf::Color::Blue, sf::Color::Cyan, sf::Color::Magenta, sf::Color::Yellow
        };
        return colorPalette[index % colorPalette.size()];
    }

void GraphWindow:: drawYAxisLabel(double value, float x, float y) {
        sf::Text label;
        label.setFont(font);
        label.setString(formatLabel(value));
        label.setCharacterSize(16);
        label.setFillColor(sf::Color::White);
        label.setPosition(x - 60, y - 10);
        window.draw(label);
    }

void GraphWindow:: drawXAxisLabel(int run, float x, float y) {
        sf::Text label;
        label.setFont(font);
        label.setString(std::to_string(run));
        label.setCharacterSize(16);
        label.setFillColor(sf::Color::White);
        label.setPosition(x - 15, y);
        window.draw(label);
    }

string GraphWindow:: formatLabel(double value) {
        std::ostringstream stream;
        stream << std::fixed << std::setprecision(2) << value;
        return stream.str();
    }
