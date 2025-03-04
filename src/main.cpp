#include "player_form.hpp"
#include "csv.hpp"
#include "team_select.hpp"
#include "PyTrain.hpp"

#define TOP_N 20

int main(int argc, char* argv[]) {

    string url;

    // Get URL from args:
    if(argc == 2) {
        url = argv[1];
    } else {
        url = URL;
    }
    url.append(BOOTSTRAP);

    cout << "Requesting player data from URL: " << url << endl;

    string playerList = fetchJSONFromURL(url);

    cout << "Creating Player Info Data Structures" << endl;

    // Create initial struct containing player information
    vector<player> playerInfo = fetchPlayers(playerList);

    cout << "Filling Player Info Data Structures" << endl;

    clearCSV();

    // Fill match history with player ratings.
    int CPV_rc = createPlayerHistory(playerInfo);

    if (CPV_rc == EXIT_FAILURE) {
        return EXIT_FAILURE;
    }

    cout << "Generating Player Form Vectors" << endl;

    // Create Vectors that can be used to train a (R)NN
    generateFinalDataset(playerInfo, CSV_DIR_OUT);

    cout << "Generating Prediction Input Data" << endl;

    // Create Vectors for the next GameWeek for each player:
    generatePredictionList(playerInfo, CSV_DIR_NEXT);


    // Run Python Script to Train Model

    int rc = runPythonTraining();
    if(rc == EXIT_FAILURE) {
        return rc;
    }

    // Print Top N Predictions

    printTopNPlayers(TOP_N, CSV_DIR_NET, playerInfo);

    // Add prediced scroes to structs

    updatePlayerScoresFromCSV(playerInfo, CSV_DIR_NET);

    // Create Top Team

    selectTeam(playerInfo, BUDGET);

    return EXIT_SUCCESS;
}
