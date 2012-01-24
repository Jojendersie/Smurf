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
	Program();
	~Program();

	// Accessors
	////////////////////////////////////////////////////////////////////////////////
	bool IsRunning() const;
	static const unsigned int& GetElapsedTime();
	static const unsigned long long& GetTotalTime();
	static float GetFramerate();

	// Public Methods
	////////////////////////////////////////////////////////////////////////////////
	void Run();
	void Exit();

private:
	// Private Methods
	////////////////////////////////////////////////////////////////////////////////
	void Initialize();
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
	GLuint smokeFBO;
	GLuint colorTex;
	GLuint depthTex;
	CudaManager* cudamanager[Globals::PROGRAM_NUM_SEEDLINES];
	GLShader* flatShader;
	GLShader* alphaShader;
	GLShader* testShader;
	GLShader* renderQuadShader;
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
	bool m_bMouseActive;
	clock_t m_timeStart,m_timeIntegrate,m_timeRender,m_normalizer;
	bool m_bStopProgram;
};


#endif // PROGRAM_HPP_
