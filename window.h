#include <vector>
#include <map>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>

#include "SFML/Graphics/Font.hpp"
#include "SFML/Graphics/RenderWindow.hpp"
#include "SFML/Window/Event.hpp"
#include "SFML/Graphics/RectangleShape.hpp"
#include "SFML/Graphics/Text.hpp"
#include "SFML/Graphics/CircleShape.hpp"

class GraphWindow {
    public:
        GraphWindow(unsigned int width, unsigned int height, const std::string& title);

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
        int mouseX = -1; // To track mouse X position
        std::map<int, std::vector<double>> dataSets;
        std::map<int, double> learningRates;
        std::map<int, sf::Color> colors;

        void drawGraph();

        void drawCursorLineWithMarkers();

        void drawAxesLabels();

        void drawAxisTitles();

        void drawKey();

        sf::Color generateColor(int index);

        void drawYAxisLabel(double value, float x, float y);

        void drawXAxisLabel(int run, float x, float y);

        std::string formatLabel(double value);
};