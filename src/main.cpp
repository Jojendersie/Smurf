/*
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Smurf
 * =====
 * ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
 *
 * Smoke Surfaces: An Interactive Flow Visualization
 * Technique Inspired by Real-World Flow Experiments
 *
 * File:              /src/main.cpp
 * Author:            Martin Kirst
 * Creation Date:     16.11.2011
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */

#include "program.hpp"
#include "amloader.hpp"

int main(int argc, char* argv[]) {
	// Amira Mesh test section ***
	AmiraMesh Mesh;
	Mesh.Load("..\\Data\\BubbleChamber_11x11x10_T0.am");
	// *** Amira Mesh test section
	return 0;
}
