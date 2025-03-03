#include <PyTrain.hpp>

#include <Python.h>

int runPythonTraining(void){
    // Initialize the Python interpreter
    Py_Initialize();
    if (!Py_IsInitialized()) {
        std::cerr << "Python initialization failed." << std::endl;
        return EXIT_FAILURE;
    }

    // Build the name of the script
    PyObject* pName = PyUnicode_DecodeFSDefault(PYTHON_SCRIPT);
    if (!pName) {
        std::cerr << "Failed to decode script name." << std::endl;
        Py_Finalize();
        return EXIT_FAILURE;
    }

    // Convert script name to UTF-8
    const char* scriptName = PyUnicode_AsUTF8(pName);
    if (!scriptName) {
        std::cerr << "Failed to convert script name to UTF-8." << std::endl;
        Py_DECREF(pName);
        Py_Finalize();
        return EXIT_FAILURE;
    }

    // Open the script file
    FILE* fp = fopen(scriptName, "r");
    if (fp == nullptr) {
        std::cerr << "Could not open script: " << scriptName << std::endl;
        Py_DECREF(pName);
        Py_Finalize();
        return EXIT_FAILURE;
    }

    // Run the script
    std::cout << "Running Python script: " << scriptName << std::endl;
    PyRun_SimpleFileEx(fp, scriptName, 1); // File will be closed by Python

    // Clean up
    Py_DECREF(pName);

    // Finalize the Python interpreter
    if (Py_FinalizeEx() < 0) {
        std::cerr << "Python finalization failed." << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}