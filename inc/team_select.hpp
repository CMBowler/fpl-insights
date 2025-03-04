#ifndef _TEAM_SELECT_HPP_
#define _TEAM_SELECT_HPP_

#include "fetch.hpp"

using namespace std;

#define BUDGET 1036

void updatePlayerScoresFromCSV(vector<player>& players, const string& csvFilePath);

void selectTeam(const vector<player>& players, int budget);

#endif // _TEAM_SELECT_HPP_
