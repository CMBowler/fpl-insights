#ifndef _PLAYER_FORM_HPP_
#define _PLAYER_FORM_HPP_

#include "score.hpp"
#include "json.hpp"

// Use the JSON library
using json = nlohmann::json;

vector<float> CalculateForm(const string& jsonContent, ROLE role);

#endif // _PLAYER_FORM_HPP_