//
// Created by Abhinav Reddy on 9/14/25.
//

#ifndef HEIGHTSMATRIX_H
#define HEIGHTSMATRIX_H

#include <iostream>
#include <vector>
#include <utility>
#include <numeric>
#include <algorithm>
#include <climits>
#include <iomanip>

class HeightsMatrix {
private:
    std::vector<std::vector<int>> heights;

    int rows, cols;
    int minVal, minRow, minCol;

    std::vector<bool> reachable;
    std::vector<int> worstDistance;

    int dr[4] = {-1, 1, 0, 0};
    int dc[4] = {0, 0, -1, 1};

    [[nodiscard]] bool isValid(int x, int y) const {
        return x >= 0 && x < rows && y >= 0 && y < cols;
    }

    void findMinimum() {
        minVal = INT_MAX;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                if (heights[i][j] < minVal) {
                    minVal = heights[i][j];
                    minRow = i;
                    minCol = j;
                }
            }
        }
    }

    int getMaximumWorstCaseDistance(const std::vector<int>& worstDistance) const {
        int maxDistance = 0;
        for ( auto it : worstDistance ) {
            maxDistance = std::max(maxDistance, it);
        }

        return maxDistance;
    }

    auto getTopoOrderOfCells() {
        int topo_len = rows * cols;

        std::vector<int> order(topo_len);
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int i, int j) {
            int ri = i/cols, ci = i%cols;
            int rj = j/cols, cj = j%cols;
            return heights[ri][ci] < heights[rj][cj];
        });

        return order;
    }

    auto idx(int r, int c) { return r * cols + c; };

public:
    explicit HeightsMatrix(std::vector<std::vector<int>>&& heightsMatrix, int rows, int cols) : heights(std::move(heightsMatrix)), rows(rows), cols(cols) {
    }

    /**
     * Used to check if the heights matrix if hilltop perfect
     * @return pair of bool (true if hilltop perfect & false otherwise)
     * and max worst distance
     */
    std::pair<bool,int> isHilltopPerfect () {
        findMinimum();

        int N = rows * cols;

        std::vector<int> order = getTopoOrderOfCells();
        reachable = std::vector<bool>(N, false);
        worstDistance = std::vector<int>(N, -1);

        reachable[idx(minRow, minCol)] = true;
        worstDistance[idx(minRow, minCol)] = 0;

        for (auto cell : order) {
            int r = cell / cols, c = cell % cols;
            if ( r == minRow && c == minCol)
                continue;

            int height = heights[r][c];

            for (int d = 0; d < 4; ++d) {
                int nr = r + dr[d], nc = c + dc[d];
                if(!isValid(nr, nc))
                    continue;

                if (reachable[idx(nr, nc)] && heights[nr][nc] < height) {
                    reachable[cell] = true;
                    worstDistance[cell] = std::max(worstDistance[cell], worstDistance[idx(nr, nc)] + 1);
                }
            }
        }

        for (auto it : reachable) if (!it) return {false, -1};

        return {true, getMaximumWorstCaseDistance(worstDistance)};

    }

    /**
     * To swap cells in the matrix
     *
     * @param x1 row of first cell
     * @param y1 column of second cell
     * @param x2 row of second cell
     * @param y2 column of second cell
     * @return
     */
    bool swapCells (int x1, int y1, int x2, int y2) {
        if (isValid(x1, y1) && isValid(x2, y2)) {
            std::swap(heights[x1][y1], heights[x2][y2]);
            return true;
        }
        return false;
    }

    void print() {
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                std::cout << std::setw(3) << heights[i][j];
                std::cout << " ";
            }
            std::cout << std::endl;
        }
    }

    void printreachable() {
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                std::cout << std::setw(3) << (reachable[idx(i, j)] ? "1" : "0");
                std::cout << " ";
            }
            std::cout << std::endl;
        }
    }

    void printworstDistance() {
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                std::cout << std::setw(3) << worstDistance[idx(i, j)];
                std::cout << " ";
            }
            std::cout << std::endl;
        }
    }
};

#endif //HEIGHTSMATRIX_H
