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
	//char* pcFile = "res\\data\\BubbleChamber_11x11x10_T0.am";
	char* pcFile = "res\\data\\SquareCylinder_192x64x48_T3048.am";
	//char* pcFile = "res\\data\\SquareCylinder\\flow_t####.am";
	//char* pcFile = "res\\data\\Wing_128x64x32_T0.am";
	bool SMOKE_TIME_DEPENDENT_INTEGRATION=false;
	unsigned int SMOKE_TIME_STEPSIZE=20;//stepsize in ms, every SMOKE_TIME_STEPSIZE ms a new slice is being taken
	unsigned int SMOKE_PARTICLE_NUMBER=500;
	float SMOKE_PRISM_THICKNESS=0.001f;
	float SMOKE_DENSITY_CONSTANT=0.005f;
	unsigned short RENDER_SMURF_ROWS=100;
	unsigned short RENDER_SMURF_COLUMS=1500;
	float SMOKE_AREA_CONSTANT_NORMALIZATION=1.f;
	float SMOKE_AREA_CONSTANT_SHARP=80.f;
	float SMOKE_SHAPE_CONSTANT=0.2f;
	float SMOKE_CURVATURE_CONSTANT=0.5f;
	float color[]={0.5f,0.5f,0.5f};

	if(argc>1) pcFile = argv[1];
	if(argc>2) SMOKE_TIME_DEPENDENT_INTEGRATION=atoi(argv[2]);
	if(argc>3) SMOKE_TIME_STEPSIZE=atoi(argv[3]);
	if(argc>4) SMOKE_PARTICLE_NUMBER=atoi(argv[4]);
	if(argc>5) SMOKE_PRISM_THICKNESS=static_cast<float>(atof(argv[5]));
	if(argc>6) SMOKE_DENSITY_CONSTANT=static_cast<float>(atof(argv[6]));
	if(argc>7) RENDER_SMURF_ROWS=atoi(argv[7]);
	if(argc>8) RENDER_SMURF_COLUMS=atoi(argv[8]);
	if(argc>9) SMOKE_AREA_CONSTANT_NORMALIZATION=static_cast<float>(atof(argv[9]));
	if(argc>10) SMOKE_AREA_CONSTANT_SHARP=static_cast<float>(atof(argv[10]));
	if(argc>11) SMOKE_SHAPE_CONSTANT=static_cast<float>(atof(argv[11]));
	if(argc>12) SMOKE_CURVATURE_CONSTANT=static_cast<float>(atof(argv[12]));
	if(argc>13) {color[0]=static_cast<float>(atof(argv[13])); color[1]=static_cast<float>(atof(argv[14])); color[2]=static_cast<float>(atof(argv[15])); }

	printf("Smurf.exe [-filename(c)] [-time dependent(ui)] [-particle_number(ui)] [-prism_thickness(f)] [-density_constant(f)] [-mesh_rows(us)] [-mesh_columns(us)] [-area_constant_normalization(f)] [-area_constant_sharp(f)] [-shape_constant(f)] [-curvature_constant(f)] [-color_r(f) -color_g(f) -color_b(f)]\n");

	printf("\n");
	printf("Smoke Surfaces (SMURF): An Interactive Flow Visualization Technique\nInspired by Real-World Flow Experiments\n\nFunctionality:\nHold the left mouse button to change the view\nWASD to move the camera\nPress the right mouse button to set one point of the seeding line\nR CPU/GPU integration\nF point/linear sampling\nN noisy/not noisy integration\nI euler/mod euler integration\n\n");

	Program program(SMOKE_TIME_DEPENDENT_INTEGRATION,
					SMOKE_TIME_STEPSIZE,
					SMOKE_PARTICLE_NUMBER,
					SMOKE_PRISM_THICKNESS,
					SMOKE_DENSITY_CONSTANT,
					RENDER_SMURF_ROWS,
					RENDER_SMURF_COLUMS,
					SMOKE_AREA_CONSTANT_NORMALIZATION,
					SMOKE_AREA_CONSTANT_SHARP,
					SMOKE_SHAPE_CONSTANT,
					SMOKE_CURVATURE_CONSTANT,
					color);

	program.Run(pcFile);
	return EXIT_SUCCESS;
}
