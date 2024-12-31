#include "csv.hpp"

#include <iostream>
#include <fstream>

void clearCSV(void) {
    ofstream outFile(CSV_OUT);
    outFile << "";
    outFile.close();
}

int exportFormToCSV(const string& filename, const vector<float>& data) {
    ofstream outFile(filename, ios::app);

    if (!outFile.is_open()) {
        cerr << "Error: Unable to open file for writing: " << filename << endl;
        return EXIT_FAILURE;
    }

    // Write data
    outFile << data[0];
    for (int i=1; i < data.size(); i++) {
        outFile << "," << data[i];
    }
    outFile << endl;

    outFile.close();

    //cout << "Form data successfully exported to " << filename << endl;

    return EXIT_SUCCESS;
}

// Function to generate the final dataset
void generateFinalDataset(const vector<player> players, const std::string& outputFilename) {
    std::ofstream outFile(outputFilename);

    if (!outFile.is_open()) {
        std::cerr << "Error: Unable to open file for writing: " << outputFilename << std::endl;
        return;
    }

    /* Write header
    outFile << "PlayerID,Cost,Team";
    for (int i = 1; i <= 5; ++i) {
        outFile << ",Match" << i << "_Rating,Match" << i << "_Opponent,Match" << i << "_Home";
    }
    outFile << ",ScoreRating\n";
    */

    // Process each player's data
    for (const auto& player : players) {
        // Ensure we have at least 5 triplets (4 for input, 1 for target)
        // & ensure the player has enough game time
        if (player.matchHist.size() < 5 || player.mins < 240) {
            continue;
        }

        // Generate sliding windows
        for (size_t start = 0; start <= player.matchHist.size() - 6; ++start) {
            outFile << player.pid << "," << player.cost << "," << player.team;

            // Write 5 triplets
            for (size_t i = 0; i < 5; ++i) {
                const auto& [opponent, home, rating] = player.matchHist[start + i];
                outFile << "," << opponent << "," << (home ? 1 : 0) << "," << rating;
            }
            outFile << endl;
        }
    }

    outFile.close();
    std::cout << "Final dataset successfully exported to " << outputFilename << std::endl;
}
