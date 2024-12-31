#include "player_form.hpp"
#include "csv.hpp"

int main(int argc, char* argv[]) {

    string url = URL;
    url.append(BOOTSTRAP);

    cout << "Requesting player data from URL: " << url << endl;

    string playerList = fetchJSONFromURL(url);

    clearCSV();

    // Create initial struct containing player information
    vector<player> playerInfo = fetchPlayers(playerList);

    // Fill match history with player ratings.
    int CPV_rc = createPlayerHistory(playerInfo);

    if (CPV_rc == EXIT_FAILURE) {
        return EXIT_FAILURE;
    }

    // Create Vectors that can be used to train a (R)NN
    generateFinalDataset(playerInfo, CSV_OUT);

    return EXIT_SUCCESS;
}
