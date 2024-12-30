#include "fetch.hpp"
#include "csv.hpp"

int main(int argc, char* argv[]) {


    string url = URL;
    url.append(BOOTSTRAP);

    cout << "Requesting player data from URL: " << url << endl;

    string playerList = fetchJSONFromURL(url);

    //startCSV();

    vector<player> playerInfo = fetchPlayers(playerList);

    for (auto player : playerInfo) {
        // Build history vector for each player

        // if a player has played a total of less
        // than 90 mins, exclude them from the dataset
        if (player.mins < 180) {
            continue;
        }

        // Build URL for player history
        string playerURL = URL;
        playerURL.append(MATCH_HISTORY);
        playerURL.append(to_string(player.pid));

        //cout    << "Requesting match history for player " << player.pid 
        //        << " from URL: " << playerURL << endl;

        string playerHist = fetchJSONFromURL(playerURL);
        vector<float> form = CalculateForm(playerHist, player.role);

        // Concatanate player data
        vector<float> output;
        output.push_back((float)player.pid);
        output.push_back((float)player.cost);
        output.insert(output.end(), form.begin(), form.end());

        // Save to csv
        int export_rc = exportFormToCSV(CSV_OUT, output);
        if (export_rc == EXIT_FAILURE) {
            return export_rc;
        }

        output.clear();
    }

    return EXIT_SUCCESS;
}
