#include "csv.hpp"

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <sstream>

void clearCSV(void) {
    ofstream outFile(CSV_DIR_OUT);
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
void generateFinalDataset(const vector<player> players, const string& outputFilename) {
    ofstream outFile(outputFilename);

    if (!outFile.is_open()) {
        cerr << "Error: Unable to open file for writing: " << outputFilename << endl;
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
        if (player.matchHist.size() < 6 || player.mins < 240) {
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
    cout << "Final dataset successfully exported to " << outputFilename << endl;
}

void generatePredictionList(const vector<player> players, const string& outputFilename){
    ofstream outFile(outputFilename);

    if (!outFile.is_open()) {
        cerr << "Error: Unable to open file for writing: " << outputFilename << endl;
        return;
    }

    // Process each player's data
    for (const auto& player : players) {
        // Ensure we have at least 5 triplets (4 for input, 1 for target)
        // & ensure the player has enough game time

        int matchesPlayed = player.matchHist.size();
        
        if (matchesPlayed < 5 || player.mins < 240) {
            continue;
        }

        outFile << player.pid << "," << player.cost << "," << player.team;

        // Write 5 triplets
        for (size_t i = matchesPlayed - 1; i > matchesPlayed - 5; --i) {
            const auto& [opponent, home, rating] = player.matchHist[i];
            outFile << "," << opponent << "," << (home ? 1 : 0) << "," << rating;
        }

        // Next fixture 
        float opponent = 0;
        bool home = true;

        string hist_url = URL;
        hist_url.append(MATCH_HISTORY);
        hist_url.append(to_string(player.pid));

        string hist_json = fetchJSONFromURL(hist_url);

        fetchNextFixture(hist_json, player, &opponent, &home);

        outFile << "," << (opponent/(float)MAX_TEAMS) << "," << home << endl;

    }

    outFile.close();
    cout << "Final dataset successfully exported to " << outputFilename << endl;
}

// Function to read the CSV file and print the top 20 player names
void printTopNPlayers(int N, const string& csvFilePath, const vector<player>& players) {
    ifstream file(csvFilePath);
    if (!file.is_open()) {
        cerr << "Could not open the CSV file." << endl;
        return;
    }

    string line;
    unordered_map<int, string> playerMap;

    // Create a map of player IDs to names for quick lookup
    for (const auto& player : players) {
        playerMap[player.pid] = player.name;
    }

    int count = 0;
    const int topN = N;

    // Discard Header
    getline(file, line);

    // Read the CSV file line by line
    while (getline(file, line) && count < topN) {
        stringstream ss(line);
        string field;
        int player_id;

        // Assuming the CSV format is: player_id,predicted_score
        getline(ss, field, ',');
        player_id = stoi(field);

        // Find the player name using the player_id
        auto it = playerMap.find(player_id);
        if (it != playerMap.end()) {
            cout << it->second << endl;
            count++;
        }
    }

    file.close();
}