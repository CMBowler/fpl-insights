#ifndef _FETCH_HPP_
#define _FETCH_HPP_

#include <string>
#include <iostream>
#include <tuple>

#include "score.hpp"

#define BOOTSTRAP "bootstrap-static/"
#define MATCH_HISTORY "element-summary/"
#define MATCHWEEK_HISTORY "fixtures/"

using namespace std;

enum TEAM {
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
    WOL,
    MAX_TEAMS
};

typedef struct player {
    int pid;
    int cost;
    ROLE role;
    int mins;
    TEAM team;
    vector<tuple<float, bool, float>> matchHist;
}player;

string fetchJSONFromURL(const string& url);

vector<player> fetchPlayers(const string& jsonContent);

#endif // _FETCH_HPP_
