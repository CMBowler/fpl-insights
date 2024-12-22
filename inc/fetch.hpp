#ifndef _FETCH_HPP_
#define _FETCH_HPP_

#include <string>
#include <iostream>

#include "form.hpp"

#define BOOTSTRAP "bootstrap-static/"
#define MATCH_HISTORY "element-summary/"

#define CSV_OUT "player_data.csv"

enum TEAMS {
    ARS = 1,
    AVL,
    BOU,
    BRE,
    BHA,
    CHE,
    CRY,
    EVE,
    FUL,
    IPS,
    LEI,
    LIV,
    MCI,
    MUN,
    NEW,
    NFO,
    SOU,
    TOT,
    WHU,
    WOL
};

using namespace std;

typedef struct player {
    int pid;
    int cost;
    ROLE role;
    int mins;
    int team;
}player;

string fetchJSONFromURL(const string& url);

vector<player> fetchPlayers(const string& jsonContent);

#endif