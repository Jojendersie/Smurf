////////////////////////////////////////////////////////////////////////////////
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


////////////////////////////////////////////////////////////////////////////////
// Preprocessor Directives and Namespaces
////////////////////////////////////////////////////////////////////////////////


#include "program.hpp"


////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
	Program program;
	char* pcFile = "res\\data\\BubbleChamber_11x11x10_T0.am";
	//char* pcFile = "res\\data\\SquareCylinder_192x64x48_T3048.am";
	if(argc>1) pcFile = argv[1];
	program.Run(pcFile);
	return EXIT_SUCCESS;
}
