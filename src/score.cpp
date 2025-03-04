#include <iostream>
#include <algorithm> // for std::mi

#include "score.hpp"

// Function to normalize a value to the range [0, 1]
static inline float normalize(float value, float max_value) {
    return min((double)value / (double)max_value, 1.0);
}

// Function to calculate the evaluation score
float calculateEvaluationScore(
    float goals, float assists, float xG, float xA,
    float bonus_points, float influence, float creativity, float threat,
    float goals_conceded, float xGC, ROLE role
) {
    // Define maximum values for normalization
    const double max_goals = 3.0;
    const double max_assists = 3.0;
    const double max_xG = 2.0;
    const double max_xA = 2.0;
    const double max_bonus_points = 75.0;
    const double max_influence = 100.0;
    const double max_creativity = 100.0;
    const double max_threat = 100.0;
    const double max_goals_conceded = 4.0; // Assume 5 goals conceded as the max significant value
    const double max_xGC = 4.0;           // Assume 5 xGC as the max significant value

    // Normalize each metric
    double normalized_goals = normalize(goals, max_goals);
    double normalized_assists = normalize(assists, max_assists);
    double normalized_xG = normalize(xG, max_xG);
    double normalized_xA = normalize(xA, max_xA);
    double normalized_bonus_points = normalize(bonus_points, max_bonus_points);
    double normalized_influence = normalize(influence, max_influence);
    double normalized_creativity = normalize(creativity, max_creativity);
    double normalized_threat = normalize(threat, max_threat);
    double normalized_goals_conceded = normalize(goals_conceded, max_goals_conceded);
    double normalized_xGC = normalize(xGC, max_xGC);

    // Define weights for each metric, dependant on the players role
    vector<float> weights;
    setWeights(weights, role);

    // Calculate weighted contributions
    double score = 0.0;
    score += weights[0] * normalized_goals;
    score += weights[1] * normalized_assists;
    score += weights[2] * (0.5 * normalized_xG + 0.5 * normalized_xA);
    score += weights[3] * normalized_bonus_points;
    score += weights[4] * (0.33 * normalized_influence + 0.33 * normalized_creativity + 0.34 * normalized_threat);
    score += weights[5] * normalized_goals_conceded; // Penalize for goals conceded
    score += weights[6] * normalized_xGC;                      // Penalize for expected goals conceded

    // Normalise the score
    score = std::max(score, 0.0); 
    score = std::min(score, 1.0); 
    
    return score;
}

/*
Test Input
int main() {
    // Example inputs
    double goals = 2.0;
    double assists = 1.0;
    double xG = 1.5;
    double xA = 1.0;
    double bonus_points = 3.0;
    double influence = 80.0;
    double creativity = 70.0;
    double threat = 90.0;
    double goals_conceded = 2.0;
    double xGC = 1.5;
    ROLE role = MIDFIELDER;

    // Calculate the evaluation score
    double evaluation_score = calculateEvaluationScore(
        goals, assists, xG, xA, bonus_points, influence, creativity, threat,
        goals_conceded, xGC, role
    );

    // Output the result
    std::cout << "Normalized Evaluation Score: " << evaluation_score << std::endl;

    return 0;
}
*/
