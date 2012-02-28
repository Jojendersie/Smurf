////////////////////////////////////////////////////////////////////////////////
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


////////////////////////////////////////////////////////////////////////////////
// Class Declaration
////////////////////////////////////////////////////////////////////////////////


class Program {
public:
	// Constructors and Destructor
	////////////////////////////////////////////////////////////////////////////////
	Program(bool SMOKE_TIME_DEPENDENT_INTEGRATION,
			unsigned int SMOKE_TIME_STEPSIZE,
			unsigned int SMOKE_PARTICLE_NUMBER,
			float SMOKE_PRISM_THICKNESS,
			float SMOKE_DENSITY_CONSTANT,
			unsigned short RENDER_SMURF_ROWS,
			unsigned short RENDER_SMURF_COLUMS,
			float SMOKE_AREA_CONSTANT_NORMALIZATION,
			float SMOKE_AREA_CONSTANT_SHARP,
			float SMOKE_SHAPE_CONSTANT,
			float SMOKE_CURVATURE_CONSTANT,
			float SMOKE_COLOR[]);
	~Program();

	// Accessors
	////////////////////////////////////////////////////////////////////////////////
	bool IsRunning() const;
	static const unsigned int& GetElapsedTime();
	static const unsigned long long& GetTotalTime();
	static float GetFramerate();

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
	static unsigned int timeCurrent;
	static unsigned int timeLast;
	static unsigned long long timeTotal;
	GLGraphics* graphics;
	GLint texLoc;
	GLuint renderQuadVAO;
	GLuint opaqueFBO, opaqueColor, opaqueDepth, smokeFBO[Globals::RENDER_DEPTH_PEELING_LAYER],colorTex[Globals::RENDER_DEPTH_PEELING_LAYER],depthTex[Globals::RENDER_DEPTH_PEELING_LAYER];
	CudaManager* cudamanager[Globals::PROGRAM_NUM_SEEDLINES];
	GLShader *flatShader, *alphaShader, *renderQuadShader, *compositingShader;
	SFCamera* camera;
	GLuint m_uiFrameCount;
	AmiraMesh m_VectorField;
	SmokeSurface* m_pSmokeSurface[Globals::PROGRAM_NUM_SEEDLINES];
	SolidSurface* m_pSolidSurface;
	unsigned int m_uiEditSeedLine;
	bool m_bCloseRequest;
	bool m_bNoisyIntegration;
	bool m_bUseLinearFilter;
	bool m_bUseAdvancedEuler;
	bool m_bUseCPUIntegration;
	bool m_bUseTimeDependentIntegration;
	bool m_bMouseActive;
	clock_t m_timeStart,m_timeIntegrate,m_timeRender,m_normalizer;
	bool m_bStopProgram;

	unsigned short RENDER_SMURF_ROWS;
	unsigned short RENDER_SMURF_COLUMS;

	float SMOKE_AREA_CONSTANT_NORMALIZATION;
	float SMOKE_AREA_CONSTANT_SHARP;
	float SMOKE_DENSITY_CONSTANT_K;
	float SMOKE_SHAPE_CONSTANT;
	float SMOKE_CURVATURE_CONSTANT;
	bool  SMOKE_TIME_DEPENDENT_INTEGRATION;
	unsigned int SMOKE_TIME_STEPSIZE;
	float SMOKE_COLOR[3];
};


#endif // PROGRAM_HPP_
