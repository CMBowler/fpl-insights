#include <stdexcept>
#include <iostream>
#include <tuple>

#include "player_form.hpp"

static float stringToFloat(const std::string& str) {
    try {
        size_t pos;
        float value = std::stof(str, &pos);

        // Check if there are invalid characters in the string
        if (pos != str.length()) {
            throw std::invalid_argument("Invalid characters found in string: " + str);
        }

        return value;
    } catch (const std::exception& e) {
        // Handle conversion errors gracefully
        std::cerr << "Error converting string to float: " << e.what() << " (input: " << str << ")" << std::endl;
        return 0.0f; // Default value if conversion fails
    }
}

// Function to parse JSON and calculate scores
void CalculateForm(player * Player, const std::string& jsonContent) {
    json matchweek_scores = json::parse(jsonContent);

    // Extract match history
    std::vector<json> recentMatches = matchweek_scores["history"];

    // Loop through each array in "history"
    for (const auto& match : recentMatches) {

        // Add the opposition team to the vector:
        float opposition = ((float)match["opponent_team"] / ((float)MAX_TEAMS-1.0));
        
        // Add if the match was home to the vector
        bool home = bool(match["was_home"]);

        if(match["minutes"] == 0) {
            Player->matchHist.emplace_back(opposition, home, 0.0f);
            continue;
        }

        float goals = match["goals_scored"];
        float assists = match["assists"];
        float xG = stringToFloat(match["expected_goals"]);
        float xA = stringToFloat(match["expected_assists"]);
        float bonus_points = match["bps"];
        float influence = stringToFloat(match["influence"]);
        float creativity = stringToFloat(match["creativity"]);
        float threat = stringToFloat(match["threat"]);
        float goals_conceded = match["goals_conceded"];
        float xGC = stringToFloat(match["expected_goals_conceded"]);

        // Calculate the evaluation score
        float score = calculateEvaluationScore(
            goals, assists, xG, xA, bonus_points, 
            influence, creativity, threat, 
            goals_conceded, xGC, Player->role
        );

        Player->matchHist.emplace_back(opposition, home, score);
    }
}

int createPlayerHistory(vector<player>& playerInfo) {

    for (auto& Player : playerInfo) {
        // Build history vector for each player

        // if a player has played a total of less
        // than 90 mins, exclude them from the dataset
        if (Player.mins < 180) {
            continue;
        }

        // Build URL for player history
        string playerURL = URL;
        playerURL.append(MATCH_HISTORY);
        playerURL.append(to_string(Player.pid));

        //cout    << "Requesting match history for player " << player.pid 
        //        << " from URL: " << playerURL << endl;

        string playerHist = fetchJSONFromURL(playerURL);

        CalculateForm(&Player, playerHist);

    }

    return EXIT_SUCCESS;

}
