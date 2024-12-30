#ifndef _TEAM_FORM_HPP_
#define _TEAM_FORM_HPP_

#include <string>

// Function to calculate the new form of a team
double calculateNewForm(double currentForm, int goalsScored, int goalsConceded, int difficulty);

// Function to calculate form progression for all teams
void calculateFormProgression(const std::string& jsonFilename);

#endif // _TEAM_FORM_HPP_