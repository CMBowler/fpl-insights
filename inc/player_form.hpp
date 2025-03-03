#ifndef _PLAYER_FORM_HPP_
#define _PLAYER_FORM_HPP_

#include "json.hpp"
#include "fetch.hpp"

// Use the JSON library
using json = nlohmann::json;

void CalculateForm(player * Player, const std::string& jsonContent);

int createPlayerHistory(vector<player>& playerInfo);

#endif // _PLAYER_FORM_HPP_
