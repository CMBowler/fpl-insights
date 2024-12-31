#ifndef _CSV_STORE_HPP_
#define _CSV_STORE_HPP_

#include <string>
#include <vector>

#include "fetch.hpp"

#define CSV_OUT "player_data.csv"

using namespace std;

void clearCSV(void);

int exportFormToCSV(const string& filename, const vector<float>& data);

void generateFinalDataset(const vector<player> players, const std::string& outputFilename);

#endif // _CSV_STORE_HPP_
