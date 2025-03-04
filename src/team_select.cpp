#include <fstream>
#include <sstream>
#include <vector>
#include <map>
#include <algorithm>

#include <team_select.hpp>

// Function to read player scores from a CSV file and update the player structs
void updatePlayerScoresFromCSV(vector<player>& players, const string& csvFilePath) {
    // Create a map to store player scores (id -> predicted_score)
    map<int, float> scoreMap;

    // Open the CSV file
    ifstream file(csvFilePath);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << csvFilePath << endl;
        return;
    }

    // Read the CSV file line by line
    string line;

    // Discard Header
    getline(file, line);

    while (getline(file, line)) {
        stringstream ss(line);
        string idStr, scoreStr;

        // Parse the line (assuming CSV format: id,predicted_score)
        if (getline(ss, idStr, ',') && getline(ss, scoreStr, ',')) {
            int id = stoi(idStr);
            float score = stof(scoreStr);

            // Add the score to the map
            scoreMap[id] = score;
        }
    }

    file.close();

    // Update the player structs with the scores from the map
    for (auto& player : players) {
        if (scoreMap.count(player.pid)) {
            player.predictedScore = scoreMap[player.pid];
        } else {
            player.predictedScore = 0;
        }
    }
}

// Function to process a group of players and generate (cost, points, selected players) tuples for selecting k players
vector<tuple<int, float, vector<int>>> processGroup(const vector<player>& group, int k, int budget) {
    vector<map<int, pair<float, vector<int>>>> dp(k + 1); // dp[i] maps cost to (points, selected players) for i players
    dp[0][0] = {0, {}}; // Base case: 0 players, 0 cost, 0 points, empty selection

    for (const auto& player : group) {
        int price = player.role;
        float points = player.predictedScore;
        int id = player.pid;

        for (int i = k - 1; i >= 0; --i) {
            for (const auto& entry : dp[i]) {
                int newCost = entry.first + price;
                float newPoints = entry.second.first + points;
                vector<int> newSelection = entry.second.second;
                newSelection.push_back(id);

                if (newCost > budget) continue; // Skip if over budget

                if (dp[i + 1].count(newCost)) {
                    if (newPoints > dp[i + 1][newCost].first) {
                        dp[i + 1][newCost] = {newPoints, newSelection};
                    }
                } else {
                    dp[i + 1][newCost] = {newPoints, newSelection};
                }
            }
        }
    }

    // Extract the entries for k players and prune
    if (dp[k].empty()) {
        return {};
    }

    vector<tuple<int, float, vector<int>>> entries;
    for (const auto& entry : dp[k]) {
        entries.emplace_back(entry.first, entry.second.first, entry.second.second);
    }

    sort(entries.begin(), entries.end()); // Sort by cost

    vector<tuple<int, float, vector<int>>> pruned;
    int maxPoints = -1;

    for (const auto& entry : entries) {
        if (get<1>(entry) > maxPoints) {
            pruned.push_back(entry);
            maxPoints = get<1>(entry);
        }
    }

    return pruned;
}

// Function to combine two pruned lists
vector<tuple<int, float, vector<int>>> combine(const vector<tuple<int, float, vector<int>>>& a, const vector<tuple<int, float, vector<int>>>& b, int budget) {
    map<int, pair<float, vector<int>>> combined;

    for (const auto& ca : a) {
        for (const auto& cb : b) {
            int totalCost = get<0>(ca) + get<0>(cb);
            if (totalCost > budget) continue; // Skip if over budget

            float totalPoints = get<1>(ca) + get<1>(cb);
            vector<int> totalSelection = get<2>(ca);
            totalSelection.insert(totalSelection.end(), get<2>(cb).begin(), get<2>(cb).end());

            if (combined.count(totalCost)) {
                if (totalPoints > combined[totalCost].first) {
                    combined[totalCost] = {totalPoints, totalSelection};
                }
            } else {
                combined[totalCost] = {totalPoints, totalSelection};
            }
        }
    }

    // Prune the combined list
    vector<tuple<int, float, vector<int>>> entries;
    for (const auto& entry : combined) {
        entries.emplace_back(entry.first, entry.second.first, entry.second.second);
    }

    sort(entries.begin(), entries.end()); // Sort by cost

    vector<tuple<int, float, vector<int>>> pruned;
    int maxPoints = -1;

    for (const auto& entry : entries) {
        if (get<1>(entry) > maxPoints) {
            pruned.push_back(entry);
            maxPoints = get<1>(entry);
        }
    }

    return pruned;
}

// Main function to select the team and print the player names
void selectTeam(const vector<player>& players, int budget) {
    // Separate players by position
    vector<player> goalkeepers, defenders, midfielders, forwards;

    for (const auto& player : players) {
        switch (player.role) {
            case GOALKEEPER: goalkeepers.push_back(player); break;
            case DEFENDER: defenders.push_back(player); break;
            case MIDFIELDER: midfielders.push_back(player); break;
            case FORWARD: forwards.push_back(player); break;
        }
    }

    // Process each group
    auto gList = processGroup(goalkeepers, 2, budget);
    auto dList = processGroup(defenders, 5, budget);
    auto mList = processGroup(midfielders, 5, budget);
    auto fList = processGroup(forwards, 3, budget);

    // Check if any group has no valid combinations
    if (gList.empty() || dList.empty() || mList.empty() || fList.empty()) {
        cout << "No valid team found within the budget." << endl;
        return;
    }

    // Combine step by step
    auto gd = combine(gList, dList, budget);
    if (gd.empty()) {
        cout << "No valid team found within the budget." << endl;
        return;
    }

    auto gdm = combine(gd, mList, budget);
    if (gdm.empty()) {
        cout << "No valid team found within the budget." << endl;
        return;
    }

    auto final = combine(gdm, fList, budget);

    // Find the maximum points within budget
    float maxPoints = -1.0;
    vector<int> bestTeam;

    for (const auto& entry : final) {
        if (get<0>(entry) <= budget && get<1>(entry) > maxPoints) {
            maxPoints = get<1>(entry);
            bestTeam = get<2>(entry);
        }
    }

    if (maxPoints == -1.0) {
        cout << "No valid team found within the budget." << endl;
        return;
    }

    // Print the selected team
    cout << "Selected Team:" << endl;
    for (int id : bestTeam) {
        for (const auto& player : players) {
            if (player.pid == id) {
                cout << "Name: " << player.name
                     << ", Position: " << player.role
                     << ", Price: " << player.cost
                     << ", Predicted Score: " << player.predictedScore << endl;
                break;
            }
        }
    }
}
