#include "csv.hpp"

#include <iostream>
#include <fstream>

int exportFormToCSV(const string& filename, const vector<float>& data) {
    ofstream outFile(filename, ios::app);

    if (!outFile.is_open()) {
        cerr << "Error: Unable to open file for writing: " << filename << endl;
        return EXIT_FAILURE;
    }

    // Write data
    outFile << data[0];
    for (int i=1; i < data.size(); i++) {
        outFile << "," << data[i];
    }
    outFile << endl;

    outFile.close();

    //cout << "Form data successfully exported to " << filename << endl;

    return EXIT_SUCCESS;
}

static void startCSV(void) {
    ofstream outFile(CSV_OUT);
    outFile << "PID, Cost, Form...\n";
    outFile.close();
}