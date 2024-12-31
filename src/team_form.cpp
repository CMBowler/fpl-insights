#include <vector>
#include "fetch.hpp"

#include "team_form.hpp"


// Function to calculate the new form of a team
double calculateNewForm(double currentForm, int goalsScored, int goalsConceded, int difficulty) {
    // Weight the result based on goals scored, goals conceded, and difficulty
    double resultImpact = (goalsScored - goalsConceded) * 10.0 / difficulty;
    return currentForm + resultImpact;
}

// Function to calculate form progression for all teams
void calculateFormProgression(const std::string& jsonFilename) {
    // Read the JSON file
    std::ifstream inFile(jsonFilename);
    if (!inFile.is_open()) {
        std::cerr << "Error: Unable to open file " << jsonFilename << std::endl;
        return;
    }

    json leagueData;
    inFile >> leagueData;
    inFile.close();

    // Initialize form map
    std::map<std::string, double> teamForms;

    // Process each matchweek
    for (const auto& matchweek : leagueData["matchweeks"]) {
        for (const auto& match : matchweek["matches"]) {
            std::string homeTeam = match["home_team"];
            std::string awayTeam = match["away_team"];
            int homeScore = match["home_score"];
            int awayScore = match["away_score"];
            int homeDifficulty = match["home_difficulty"];
            int awayDifficulty = match["away_difficulty"];

            // Get current form or initialize to 0 if not present
            double homeForm = teamForms.count(homeTeam) ? teamForms[homeTeam] : 0.0;
            double awayForm = teamForms.count(awayTeam) ? teamForms[awayTeam] : 0.0;

            // Calculate new forms
            double newHomeForm = calculateNewForm(homeForm, homeScore, awayScore, homeDifficulty);
            double newAwayForm = calculateNewForm(awayForm, awayScore, homeScore, awayDifficulty);

            // Update the forms
            teamForms[homeTeam] = newHomeForm;
            teamForms[awayTeam] = newAwayForm;
        }
    }

    // Print the final forms
    for (const auto& [team, form] : teamForms) {
        std::cout << "Team: " << team << ", Final Form: " << form << std::endl;
    }
}
