#ifndef _CSV_STORE_HPP_
#define _CSV_STORE_HPP_

#include <string>
#include <vector>

#define CSV_OUT "player_data.csv"

using namespace std;

int exportFormToCSV(const string& filename, const vector<float>& data);


#endif // _CSV_STORE_HPP_