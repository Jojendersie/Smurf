﻿////////////////////////////////////////////////////////////////////////////////
//
// Smurf
// =====
// ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
//
// Smoke Surfaces: An Interactive Flow Visualization
// Technique Inspired by Real-World Flow Experiments
//
// File:              /include/program.hpp
// Author:            Martin Kirst
// Creation Date:     2011.11.16
// Description:
//
// Declaration of the main program instance.
// The main program instance is responsible for creating the window, managing the runtime behavior,
// controlling the main program sequence, reserving the render context
// and running the main execution loop.
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Preprocessor Directives and Namespaces
////////////////////////////////////////////////////////////////////////////////


#ifndef PROGRAM_HPP_
#define PROGRAM_HPP_

#include <time.h>
#include <SFML/Window.hpp>
#include "glgraphics.hpp"
#include "glshader.hpp"
#include "sfcamera.hpp"
#include "amloader.hpp"
#include "smokesurface.hpp"
#include "solidgeometry.hpp"
#include "cudamanager.hpp"

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
extern float RENDER_SMURF_STEPSIZE;
extern unsigned int RENDER_POLYGON_MAX;

////////////////////////////////////////////////////////////////////////////////
// Class Declaration
////////////////////////////////////////////////////////////////////////////////


class Program {
public:
	// Constructors and Destructor
	////////////////////////////////////////////////////////////////////////////////
	Program();
	~Program();

	// Accessors
	////////////////////////////////////////////////////////////////////////////////
	bool IsRunning() const;
	const unsigned int& GetElapsedTime();
	const unsigned long long& GetTotalTime();
	float GetFramerate();

	// Public Methods
	////////////////////////////////////////////////////////////////////////////////
	void Run(const char* _pcFile);
	void Exit();

private:
	// Private Methods
	////////////////////////////////////////////////////////////////////////////////
	void Initialize(const char* _pcFile);
	void Update();
	void Draw();
	void HandleBasicEvents();
	void RayCast();

	// Variables
	////////////////////////////////////////////////////////////////////////////////
	sf::Window mainWindow;
	sf::Clock m_clock;
	unsigned int timeCurrent;
	unsigned int timeLast;
	unsigned long long timeTotal;
	GLGraphics* graphics;
	GLint texLoc;
	GLuint renderQuadVAO;
	GLuint opaqueFBO, opaqueColor, opaqueDepth, *smokeFBO,*colorTex,*depthTex;
	CudaManager **cudamanager;
	GLShader *flatShader, *alphaShader, *renderQuadShader, *compositingShader;
	SFCamera* camera;
	GLuint m_uiFrameCount;
	AmiraMesh m_VectorField;
	SmokeSurface **m_pSmokeSurface;
	SolidSurface* m_pSolidSurface;
	unsigned int m_uiEditSeedLine;
	bool m_bWireframe;
	bool m_bCloseRequest;
	bool m_bNoisyIntegration;
	bool m_bUseLinearFilter;
	bool m_bUseAdvancedEuler;
	bool m_bUseCPUIntegration;
	bool m_bMouseActive;
	clock_t m_timeStart,m_timeIntegrate,m_timeRender,m_normalizer;
	bool m_bStopProgram;
};


#endif // PROGRAM_HPP_
