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

#include <iostream>
#include <fstream>
#include <string>

#include "program.hpp"
#include "parameter.hpp"

void LoadParameters(const char *filename);

extern char* pcFile;
extern bool SMOKE_TIME_DEPENDENT_INTEGRATION;
extern unsigned int SMOKE_TIME_STEPSIZE;
extern unsigned int SMOKE_PARTICLE_NUMBER;
extern float SMOKE_PRISM_THICKNESS;
extern float SMOKE_DENSITY_CONSTANT;
extern float SMOKE_DENSITY_CONSTANT_K;
extern unsigned short RENDER_SMURF_ROWS;
extern unsigned short RENDER_SMURF_COLUMS;
extern float SMOKE_AREA_CONSTANT_NORMALIZATION;
extern float SMOKE_SHAPE_CONSTANT;
extern float SMOKE_CURVATURE_CONSTANT;
extern float SMOKE_COLOR[];

extern unsigned short PROGRAM_FRAMES_PER_RELEASE;
extern unsigned short PROGRAM_NUM_SEEDLINES;
extern unsigned short RENDER_DEPTH_PEELING_LAYER;

////////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[]) {
	//if(argc>1) pcFile = argv[1];
	//if(argc>2) SMOKE_TIME_DEPENDENT_INTEGRATION=atoi(argv[2]);
	//if(argc>3) SMOKE_TIME_STEPSIZE=atoi(argv[3]);
	//if(argc>4) SMOKE_PARTICLE_NUMBER=atoi(argv[4]);
	//if(argc>5) SMOKE_PRISM_THICKNESS=static_cast<float>(atof(argv[5]));
	//if(argc>6) SMOKE_DENSITY_CONSTANT=static_cast<float>(atof(argv[6]));
	//if(argc>7) RENDER_SMURF_ROWS=atoi(argv[7]);
	//if(argc>8) RENDER_SMURF_COLUMS=atoi(argv[8]);
	//if(argc>9) SMOKE_AREA_CONSTANT_NORMALIZATION=static_cast<float>(atof(argv[9]));
	//if(argc>10) SMOKE_SHAPE_CONSTANT=static_cast<float>(atof(argv[11]));
	//if(argc>11) SMOKE_CURVATURE_CONSTANT=static_cast<float>(atof(argv[12]));
	//if(argc>12) {SMOKE_COLOR[0]=static_cast<float>(atof(argv[13])); SMOKE_COLOR[1]=static_cast<float>(atof(argv[14])); SMOKE_COLOR[2]=static_cast<float>(atof(argv[15]));}

	//SMOKE_DENSITY_CONSTANT_K=SMOKE_PRISM_THICKNESS*SMOKE_PARTICLE_NUMBER*SMOKE_DENSITY_CONSTANT;

	printf("Smurf.exe [-filename(c)] [-time dependent(ui)] [-time stepsize(ui)] [-particle_number(ui)] [-prism_thickness(f)] [-density_constant(f)] [-mesh_rows(us)] [-mesh_columns(us)] [-area_constant_normalization(f)] [-shape_constant(f)] [-curvature_constant(f)] [-color_r(f) -color_g(f) -color_b(f)]\n");

	printf("\n");
	printf("Smoke Surfaces (SMURF): An Interactive Flow Visualization Technique\nInspired by Real-World Flow Experiments\n\nFunctionality:\nHold the left mouse button to change the view\nWASD to move the camera\nPress the right mouse button to set one point of the seeding line\nR CPU/GPU integration\nF point/linear sampling\nN noisy/not noisy integration\nI euler/mod euler integration\n\n");

	LoadParameters("res\\parameters.cfg");

	Program program;

	program.Run(pcFile);

	return EXIT_SUCCESS;
}

void LoadParameters(const char *filename)
{
	std::string line;
	std::fstream file(filename,std::ios::in);

	std::getline(file,line,' ');	
	std::getline(file,line);

	pcFile = new char[line.length()];
	std::sprintf(pcFile,"%s",line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	SMOKE_TIME_DEPENDENT_INTEGRATION=atoi(line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	SMOKE_TIME_STEPSIZE=atoi(line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	SMOKE_PARTICLE_NUMBER=atoi(line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	SMOKE_PRISM_THICKNESS=atof(line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	SMOKE_DENSITY_CONSTANT=atof(line.c_str());
	SMOKE_DENSITY_CONSTANT_K=SMOKE_PRISM_THICKNESS*SMOKE_PARTICLE_NUMBER*SMOKE_DENSITY_CONSTANT;

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	RENDER_SMURF_ROWS=atoi(line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	RENDER_SMURF_COLUMS=atoi(line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	SMOKE_AREA_CONSTANT_NORMALIZATION=atof(line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	SMOKE_SHAPE_CONSTANT=atof(line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	SMOKE_CURVATURE_CONSTANT=atof(line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	SMOKE_COLOR[0]=atof(line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	SMOKE_COLOR[1]=atof(line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	SMOKE_COLOR[2]=atof(line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	PROGRAM_FRAMES_PER_RELEASE=atoi(line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	PROGRAM_NUM_SEEDLINES=atoi(line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	RENDER_DEPTH_PEELING_LAYER=atoi(line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	RENDER_SMURF_STEPSIZE=atof(line.c_str());

	std::getline(file,line,' ');
	std::getline(file,line);
	line+='\n';
	RENDER_POLYGON_MAX=atoi(line.c_str());

	file.close();

	printf(pcFile);
}