#ifndef _CSV_STORE_HPP_
#define _CSV_STORE_HPP_

#include <string>
#include <vector>

#include "fetch.hpp"

#define CSV_DIR "../../csv/"

#define CSV_OUT "player_data.csv"
#define CSV_NEXT "player_fixture.csv"
#define CSV_NET "predictions.csv"

#define CSV_DIR_OUT CSV_DIR CSV_OUT
#define CSV_DIR_NEXT CSV_DIR CSV_NEXT
#define CSV_DIR_NET CSV_DIR CSV_NET

using namespace std;

void clearCSV(void);

int exportFormToCSV(const string& filename, const vector<float>& data);

void generateFinalDataset(const vector<player> players, const string& outputFilename);

void generatePredictionList(const vector<player> players, const string& outputFilename);

// Function to read the CSV file and print the top 20 player names
void printTopNPlayers(int N, const string& csvFilePath, const vector<player>& players);

#endif // _CSV_STORE_HPP_
