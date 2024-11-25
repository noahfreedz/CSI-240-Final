#include "neural_network.h"
#include <stdexcept>
#include <limits>
#include <algorithm>

using namespace std;
using namespace Rebecca;

GraphWindow::GraphWindow(unsigned int width, unsigned int height, const string& title, ThreadNetworks* _allNetworks)
    : window(sf::VideoMode(width, height), title),
      window_width(static_cast<float>(width)),
      window_height(static_cast<float>(height)),
      allNetworks(_allNetworks)
{
    if (!_allNetworks) {
        throw std::runtime_error("Invalid ThreadNetworks pointer");
    }

    window.setFramerateLimit(60);

    // Enhanced font loading with error checking
    if (!font.loadFromFile("Font.ttf")) {
        std::cerr << "Failed to load font\n";
        throw std::runtime_error("Failed to load required font file: Font.ttf");
    }

    isInitialized = true;
}

GraphWindow::~GraphWindow() {
    // Ensure clean shutdown
    if (window.isOpen()) {
        window.close();
    }
}

bool GraphWindow::isOpen() const {
    return window.isOpen() && isInitialized;
}

void GraphWindow::handleEvents() {
    if (!isOpen()) return;

    sf::Event event;
    while (window.pollEvent(event)) {
        switch (event.type) {
            case sf::Event::Closed:
                window.close();
                run_network = false;
                break;
            case sf::Event::MouseMoved:
                mouseX = event.mouseMove.x;
                break;
            case sf::Event::KeyPressed:
                if (event.key.code == sf::Keyboard::Escape) {
                    window.close();
                    run_network = false;
                }
                break;
            default:
                break;
        }
    }
}

bool GraphWindow::validateDataPoint(double value) const {
    return std::isfinite(value) &&
           value > -std::numeric_limits<double>::max() &&
           value < std::numeric_limits<double>::max();
}

void GraphWindow::addDataPoint(int lineID, double value) {
    if (!validateDataPoint(value)) return;

    std::lock_guard<std::mutex> lock(dataMutex);
    try {
        if (dataSets.find(lineID) == dataSets.end()) {
            dataSets[lineID] = std::vector<double>();
            colors[lineID] = generateColor(lineID);
        }
        dataSets[lineID].push_back(value);
    } catch (const std::exception& e) {
        std::cerr << "Error adding data point: " << e.what() << std::endl;
    }
}

void GraphWindow::setLearningRate(int lineID, double learningRate) {
    if (!validateDataPoint(learningRate)) return;

    std::lock_guard<std::mutex> lock(dataMutex);
    learningRates[lineID] = learningRate;
}

void GraphWindow::render() {
    if (!isOpen()) return;

    std::lock_guard<std::mutex> lock(dataMutex);

    try {
        window.clear(sf::Color::Black);

        if (!dataSets.empty()) {
            drawGraph();
            drawAxesLabels();
            drawAxisTitles();
            drawKey();
            drawCursorLineWithMarkers();
        }

        window.display();
    } catch (const std::exception& e) {
        std::cerr << "Render error: " << e.what() << std::endl;
    }
}

void GraphWindow::drawGraph() {
    if (dataSets.empty()) return;

    try {
        // Calculate global min/max safely
        double globalMax = -std::numeric_limits<double>::infinity();
        double globalMin = std::numeric_limits<double>::infinity();

        for (const auto& [id, data] : dataSets) {
            if (!data.empty()) {
                auto [min, max] = std::minmax_element(
                    data.begin(),
                    data.end(),
                    [](double a, double b) { return std::isfinite(a) && std::isfinite(b) && a < b; }
                );
                if (min != data.end() && max != data.end() &&
                    std::isfinite(*min) && std::isfinite(*max)) {
                    globalMin = std::min(globalMin, *min);
                    globalMax = std::max(globalMax, *max);
                }
            }
        }

        if (!std::isfinite(globalMin) || !std::isfinite(globalMax)) return;
        if (std::abs(globalMax - globalMin) < 1e-10) {
            globalMax = globalMin + 1.0; // Prevent division by zero
        }

        // Draw axes
        sf::Vertex xAxis[] = {
            sf::Vertex(sf::Vector2f(100.f, window_height - 80.f)),
            sf::Vertex(sf::Vector2f(window_width - 80.f, window_height - 80.f))
        };

        sf::Vertex yAxis[] = {
            sf::Vertex(sf::Vector2f(100.f, 50.f)),
            sf::Vertex(sf::Vector2f(100.f, window_height - 80.f))
        };

        window.draw(xAxis, 2, sf::Lines);
        window.draw(yAxis, 2, sf::Lines);

        // Draw data lines
        float yScale = (window_height - 160.f) / (globalMax - globalMin);

        for (const auto& [id, data] : dataSets) {
            if (data.empty()) continue;

            float xSpacing = (window_width - 180.f) / std::max(1.0f, static_cast<float>(data.size() - 1));
            std::vector<sf::Vertex> points;
            points.reserve(data.size());

            for (size_t i = 0; i < data.size(); ++i) {
                if (std::isfinite(data[i])) {
                    float x = 100.f + static_cast<float>(i) * xSpacing;
                    float y = window_height - 80.f - static_cast<float>((data[i] - globalMin) * yScale);

                    if (std::isfinite(x) && std::isfinite(y)) {
                        points.emplace_back(sf::Vector2f(x, y), colors[id]);
                    }
                }
            }

            if (points.size() >= 2) {
                window.draw(&points[0], points.size(), sf::LinesStrip);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error drawing graph: " << e.what() << std::endl;
    }
}

void GraphWindow::drawCursorLineWithMarkers() {
    if (mouseX < 100 || mouseX > window_width - 80) return;

    try {
        // Draw cursor line
        sf::Vertex cursorLine[] = {
            sf::Vertex(sf::Vector2f(static_cast<float>(mouseX), 50.f), sf::Color(200, 200, 200)),
            sf::Vertex(sf::Vector2f(static_cast<float>(mouseX), window_height - 80.f), sf::Color(200, 200, 200))
        };
        window.draw(cursorLine, 2, sf::Lines);

        if (dataSets.empty()) return;

        // Calculate scales and indices safely
        double globalMax = -std::numeric_limits<double>::infinity();
        double globalMin = std::numeric_limits<double>::infinity();

        for (const auto& [id, data] : dataSets) {
            if (!data.empty()) {
                auto [min, max] = std::minmax_element(data.begin(), data.end());
                if (min != data.end() && max != data.end()) {
                    globalMin = std::min(globalMin, *min);
                    globalMax = std::max(globalMax, *max);
                }
            }
        }

        if (!std::isfinite(globalMin) || !std::isfinite(globalMax)) return;

        double yScale = (window_height - 160.0) / (globalMax - globalMin);
        auto firstDataSet = dataSets.begin()->second;
        if (firstDataSet.empty()) return;

        double xSpacing = (window_width - 180.0) / static_cast<double>(firstDataSet.size());
        int index = static_cast<int>((mouseX - 100) / xSpacing);

        if (index < 0 || index >= static_cast<int>(firstDataSet.size())) return;

        // Draw markers and labels
        for (const auto& [id, data] : dataSets) {
            if (static_cast<size_t>(index) < data.size()) {
                double x = 100.0 + index * xSpacing;
                double y = window_height - 80.0 - (data[index] - globalMin) * yScale;

                if (std::isfinite(x) && std::isfinite(y)) {
                    // Draw marker
                    sf::CircleShape marker(3.f);
                    marker.setPosition(static_cast<float>(x - 3), static_cast<float>(y - 3));
                    marker.setFillColor(colors[id]);
                    window.draw(marker);

                    // Draw label
                    sf::Text markerText;
                    markerText.setFont(font);
                    markerText.setString("Run: " + std::to_string(perRunCount * index) +
                                      "\nCost: " + formatLabel(data[index]));
                    markerText.setCharacterSize(24);
                    markerText.setFillColor(colors[id]);
                    markerText.setPosition(static_cast<float>(x + 5), static_cast<float>(y - 15));
                    window.draw(markerText);
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error drawing cursor markers: " << e.what() << std::endl;
    }
}

void GraphWindow::drawAxesLabels() {
    if (dataSets.empty()) return;

    try {
        // Calculate value ranges
        double globalMax = -std::numeric_limits<double>::infinity();
        double globalMin = std::numeric_limits<double>::infinity();
        size_t maxDataSize = 0;

        for (const auto& [id, data] : dataSets) {
            if (!data.empty()) {
                auto [min, max] = std::minmax_element(data.begin(), data.end());
                if (min != data.end() && max != data.end()) {
                    globalMin = std::min(globalMin, *min);
                    globalMax = std::max(globalMax, *max);
                }
                maxDataSize = std::max(maxDataSize, data.size());
            }
        }

        if (!std::isfinite(globalMin) || !std::isfinite(globalMax)) return;

        // Draw Y-axis labels
        double range = globalMax - globalMin;
        std::vector<double> yValues = {
            globalMin,
            globalMin + range * 0.25,
            globalMin + range * 0.5,
            globalMin + range * 0.75,
            globalMax
        };

        float yScale = (window_height - 160.f) / range;
        for (double value : yValues) {
            float y = window_height - 80.f - static_cast<float>((value - globalMin) * yScale);
            drawYAxisLabel(value, 100.f, y);
        }

        // Draw X-axis labels
        if (maxDataSize > 1) {
            size_t numLabels = std::min<size_t>(6, maxDataSize);
            float xSpacing = (window_width - 180.f) / static_cast<float>(maxDataSize - 1);

            for (size_t i = 0; i < numLabels; ++i) {
                size_t index = i * (maxDataSize - 1) / (numLabels - 1);
                float x = 100.f + static_cast<float>(index) * xSpacing;
                drawXAxisLabel(perRunCount * static_cast<int>(index), x, window_height - 70.f);
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Error drawing axes labels: " << e.what() << std::endl;
    }
}

void GraphWindow::drawAxisTitles() {
    try {
        sf::Text yAxisTitle;
        yAxisTitle.setFont(font);
        yAxisTitle.setString("COST");
        yAxisTitle.setCharacterSize(20);
        yAxisTitle.setFillColor(sf::Color::White);
        yAxisTitle.setPosition(10.f, (window_height / 2.f) - (yAxisTitle.getLocalBounds().width / 2.f));
        yAxisTitle.setRotation(-90.f);
        window.draw(yAxisTitle);

        sf::Text xAxisTitle;
        xAxisTitle.setFont(font);
        xAxisTitle.setString("RUNS");
        xAxisTitle.setCharacterSize(20);
        xAxisTitle.setFillColor(sf::Color::White);
        xAxisTitle.setPosition((window_width / 2.f) - 30.f, window_height - 40.f);
        window.draw(xAxisTitle);
    } catch (const std::exception& e) {
        std::cerr << "Error drawing axis titles: " << e.what() << std::endl;
    }
}
void GraphWindow::drawKey() {
    try {
        sf::Text keyTitle;
        keyTitle.setFont(font);
        keyTitle.setString("Learning Rates");
        keyTitle.setCharacterSize(16);
        keyTitle.setFillColor(sf::Color::White);
        keyTitle.setPosition(window_width - 180.f, 50.f);
        window.draw(keyTitle);

        int lineCount = 0;
        for (const auto& [id, rate] : learningRates) {
            if (!std::isfinite(rate)) continue;

            sf::RectangleShape colorBox(sf::Vector2f(15.f, 15.f));
            colorBox.setFillColor(colors[id]);
            colorBox.setPosition(window_width - 180.f, 80.f + lineCount * 30.f);
            window.draw(colorBox);

            sf::Text labelText;
            labelText.setFont(font);
            labelText.setString("LR = " + formatLabel(rate));
            labelText.setCharacterSize(14);
            labelText.setFillColor(sf::Color::White);
            labelText.setPosition(window_width - 155.f, 78.f + lineCount * 30.f);
            window.draw(labelText);

            lineCount++;
        }
    } catch (const std::exception& e) {
        std::cerr << "Error drawing key: " << e.what() << std::endl;
    }
}

sf::Color GraphWindow::generateColor(int index) {
    try {
        static const std::vector<sf::Color> colorPalette = {
            sf::Color::Red,
            sf::Color::Green,
            sf::Color::Blue,
            sf::Color::Cyan,
            sf::Color::Magenta,
            sf::Color::Yellow,
            sf::Color(255, 165, 0),   // Orange
            sf::Color(128, 0, 128),   // Purple
            sf::Color(165, 42, 42),   // Brown
            sf::Color(255, 192, 203)  // Pink
        };

        return colorPalette[static_cast<size_t>(index) % colorPalette.size()];
    } catch (const std::exception& e) {
        std::cerr << "Error generating color: " << e.what() << std::endl;
        return sf::Color::White; // Default fallback color
    }
}

void GraphWindow::drawYAxisLabel(double value, float x, float y) {
    if (!std::isfinite(value) || !std::isfinite(x) || !std::isfinite(y)) return;

    try {
        sf::Text label;
        label.setFont(font);
        label.setString(formatLabel(value));
        label.setCharacterSize(16);
        label.setFillColor(sf::Color::White);

        // Adjust position to center the text vertically
        float textHeight = label.getLocalBounds().height;
        label.setPosition(x - 60.f, y - (textHeight / 2.f));

        window.draw(label);
    } catch (const std::exception& e) {
        std::cerr << "Error drawing Y axis label: " << e.what() << std::endl;
    }
}

void GraphWindow::drawXAxisLabel(int run, float x, float y) {
    if (!std::isfinite(x) || !std::isfinite(y)) return;

    try {
        sf::Text label;
        label.setFont(font);
        label.setString(std::to_string(run));
        label.setCharacterSize(16);
        label.setFillColor(sf::Color::White);

        // Center the text horizontally under the axis point
        float textWidth = label.getLocalBounds().width;
        label.setPosition(x - (textWidth / 2.f), y);

        window.draw(label);
    } catch (const std::exception& e) {
        std::cerr << "Error drawing X axis label: " << e.what() << std::endl;
    }
}

std::string GraphWindow::formatLabel(double value) {
    try {
        if (!std::isfinite(value)) return "N/A";

        std::ostringstream stream;
        stream << std::fixed << std::setprecision(2);

        if (std::abs(value) < 0.01) {
            stream << std::scientific;
        }

        stream << value;
        return stream.str();
    } catch (const std::exception& e) {
        std::cerr << "Error formatting label: " << e.what() << std::endl;
        return "Error";
    }
}