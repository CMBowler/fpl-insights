#include <sstream>
#include <iostream>
#include <curl/curl.h>

#include "json.hpp"
#include "fetch.hpp"
#include "player_form.hpp"

static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    size_t totalSize = size * nmemb;
    userp->append((char*)contents, totalSize);
    return totalSize;
}

// Function to fetch JSON data from a URL
string fetchJSONFromURL(const string& url) {
    CURL* curl;
    CURLcode res;
    string response;

    curl = curl_easy_init();
    if (curl) {
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
        curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L); // Handle redirects
        res = curl_easy_perform(curl);
        if (res != CURLE_OK) {
            throw runtime_error("cURL failed with error: " + string(curl_easy_strerror(res)));
        }
        curl_easy_cleanup(curl);
    } else {
        throw runtime_error("Failed to initialize cURL.");
    }

    return response;
}

vector<player> fetchPlayers(const string& jsonContent) {
    json matchweek_scores = json::parse(jsonContent);

    // Extract elements:
    vector<json> playerInfo = matchweek_scores["elements"];

    vector<player> output;

    // Loop through each player in playerinfo
    for(const auto info : playerInfo) {
        player newPlayer;

        newPlayer.pid = info["id"];
        newPlayer.cost = info["now_cost"];
        newPlayer.mins = info["minutes"];
        newPlayer.role = info["element_type"];
        newPlayer.team = info["team"];
        newPlayer.name = info["web_name"];

        output.push_back(newPlayer);
    }

    return output;
}

// Function to fetch the details of the next fixture for a given player
void fetchNextFixture(const string& jsonContent, const player Player, float * opponent, bool * home) {
    // Parse the JSON content to get the matchweek fixtures
    json matchweek_fixtures = json::parse(jsonContent);

    // Get the first fixture from the matchweek fixtures
    json next_fixture = matchweek_fixtures["fixtures"][0];

    // Determine the opponent and whether it is a home or away game based on the player's team and the next fixture's details
    *opponent = (((int)(Player.team) == next_fixture["team_h"]) ? 
        (float)next_fixture["team_a"] : (float)next_fixture["team_h"]
    );
    
    // Set whether it is a home or away game based on the player's team and the next fixture's details
    *home = ((int)(Player.team) == next_fixture["team_h"]) ? true : false;
}
