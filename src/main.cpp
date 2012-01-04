//
// Smurf
// =====
// ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
//
// Smoke Surfaces: An Interactive Flow Visualization
// Technique Inspired by Real-World Flow Experiments
//
// File:              /src/main.cpp
// Author:            Martin Kirst
// Creation Date:     2011.11.16
// Description:
//
// Startup file of the program.
//
////////////////////////////////////////////////////////////////////////////////

// Preprocessor Directives and Namespaces
////////////////////////////////////////////////////////////////////////////////
#include "program.hpp"

// Main C/C++ Runtime Startup (Entry Point: mainCRTStartup)
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
	Program program;
	program.Run();
	return EXIT_SUCCESS;
}
