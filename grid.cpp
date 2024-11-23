//
// Created by Kele Fireheart on 11/15/24.
//
#include "grid.h"

Grid::Grid() : gridOfSquares(gridSize, vector<double>(gridSize, 0.0)),
               colorGrid(gridSize, vector<Color>(gridSize, WHITE)){

    InitWindow(SCREEN_WIDTH, SCREEN_HEIGHT, "Grid");
    SetTargetFPS(120);
    HideCursor();

    Rectangle eraseButton = {SCREEN_WIDTH - 150, 10, 140, 50};
    Rectangle doneButton = {SCREEN_WIDTH - 150, 100, 140, 50};

    while (!WindowShouldClose()) {
        cursorPos = GetMousePosition();

        if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
            calculateDrawing(cursorPos, gridOfSquares, colorGrid, gridSize, cellSize, startX, startY);
        }
        if (isEraseButtonClicked(eraseButton)) {
            eraseGrid(gridOfSquares, colorGrid, gridSize);
        }
        if (isDoneButtonClicked(doneButton)) {
            ShowCursor();
            break;  // Exit loop when "Done" is clicked
        }

        BeginDrawing();
        ClearBackground(BLACK);

        for (int row = 0; row < gridSize; row++) {
            for (int col = 0; col < gridSize; col++) {
                int cellX = startX + col * cellSize;
                int cellY = startY + row * cellSize;

                DrawRectangle(cellX, cellY, cellSize, cellSize, colorGrid[row][col]);
                DrawRectangleLines(cellX, cellY, cellSize, cellSize, BLACK);
            }
        }

        drawEraseButton(eraseButton, "Erase All");
        drawEnterButton(doneButton, "Done");
        DrawCircleV(GetMousePosition(), 10, cursorColor);

        EndDrawing();
    }

    CloseWindow();
}

vector<vector<double>> Grid::getGridOfSquares() const {
    return gridOfSquares;
}
bool Grid::AreColorsEqual(Color c1, Color c2) {
    return c1.r == c2.r && c1.g == c2.g && c1.b == c2.b && c1.a == c2.a;
}

Color Grid::generateRandomGray() {
    int grayValue = GetRandomValue(80, 200);
    return Color{(unsigned char)grayValue, (unsigned char)grayValue, (unsigned char)grayValue, 255};
}

void Grid::calculateDrawing(const Vector2& mousePos, vector<vector<double>>& grid, vector<vector<Color>>& colorGrid,
                      int gridSize, int cellSize,
                      int startX, int startY) {

    if (IsMouseButtonDown(MOUSE_BUTTON_LEFT)) {
        Vector2 mousePos = GetMousePosition();

        if (mousePos.x >= startX && mousePos.x < startX + gridWidth &&
            mousePos.y >= startY && mousePos.y < startY + gridHeight) {

            int col = (mousePos.x - startX) / cellSize;
            int row = (mousePos.y - startY) / cellSize;

            if (!AreColorsEqual(colorGrid[row][col], BLACK)) {
                grid[row][col] = 1.0; // Example usage of double type
                colorGrid[row][col] = BLACK; // Center cell
            }

            const int neighborOffsets[4][2] = {
                    {0, -1}, {0,  1}, {-1, 0}, {1,  0}
            };

            for (const auto& offset : neighborOffsets) {
                int neighborRow = row + offset[1];
                int neighborCol = col + offset[0];

                if (neighborRow >= 0 && neighborRow < gridSize &&
                    neighborCol >= 0 && neighborCol < gridSize) {

                    if (!AreColorsEqual(colorGrid[neighborRow][neighborCol], BLACK)) {
                        grid[neighborRow][neighborCol] = 0.5; // Example usage for neighboring cells
                        colorGrid[neighborRow][neighborCol] = generateRandomGray();
                    }
                }
            }
        }
    }
}

void Grid::eraseGrid(vector<vector<double>>& grid, vector<vector<Color>>& colorGrid, int gridSize) {
    for (int row = 0; row < gridSize; ++row) {
        for (int col = 0; col < gridSize; ++col) {
            grid[row][col] = 0.0;
            colorGrid[row][col] = WHITE;
        }
    }
}

void Grid::drawEraseButton(Rectangle buttonRect, const char* buttonText) {
    DrawRectangleRec(buttonRect, GRAY);
    DrawText(buttonText, buttonRect.x + 10, buttonRect.y + 10, 20, BLACK);
}

void Grid::drawEnterButton(Rectangle buttonRect, const char* buttonText){
    DrawRectangleRec(buttonRect, GREEN);
    DrawText(buttonText, buttonRect.x + 10, buttonRect.y + 10, 20, BLACK);
}

bool Grid::isEraseButtonClicked(Rectangle buttonRect) {
    return IsMouseButtonPressed(MOUSE_BUTTON_LEFT) &&
           CheckCollisionPointRec(GetMousePosition(), buttonRect);
}

bool Grid::isDoneButtonClicked(Rectangle buttonRect){
    return IsMouseButtonPressed(MOUSE_BUTTON_LEFT) &&
           CheckCollisionPointRec(GetMousePosition(), buttonRect);
}

