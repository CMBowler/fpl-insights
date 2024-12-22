#include <stdexcept>
#include <iostream>

#include "form.hpp"

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
std::vector<float> CalculateForm(const std::string& jsonContent, ROLE role) {
    json matchweek_scores = json::parse(jsonContent);

    // Extract match history
    std::vector<json> recentMatches = matchweek_scores["history"];

    std::vector<float> all_scores;

    // Loop through each array in "history"
    for (const auto& match : recentMatches) {

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
            goals_conceded, xGC, role
        );

        // Store the matchweek scores in the main vector
        all_scores.push_back(score);
    }

    return all_scores;
}
