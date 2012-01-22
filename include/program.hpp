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

	// Variables
	////////////////////////////////////////////////////////////////////////////////
	sf::Window mainWindow;
	static unsigned int timeCurrent;
	static unsigned int timeLast;
	static unsigned long long timeTotal;
	GLGraphics* graphics;
	GLint texLoc;
	CudaManager cudamanager;
	GLShader* flatShader;
	GLShader* alphaShader;
	GLShader* testShader;
	SFCamera* camera;
	GLuint m_uiFrameCount;
	AmiraMesh m_VectorField;
	SmokeSurface* m_pSmokeSurface;
	SolidSurface* m_pSolidSurface;
	bool m_bCloseRequest;
	bool m_bNoisyIntegration;
	bool m_bUseLinearFilter;
	bool m_bUseAdvancedEuler;

	// TODO @Martin
	// If you click once (the moment where m_bInvalidSeedLine==false) -> {m_pSmokeSurface->SetSeedLineStart(m_VectorField.RayCast(...)); m_bInvalidSeedLine=true;}
	// If clicked second time (m_bInvalidSeedLine==true) -> {m_pSmokeSurface->SetSeedLineEnd(m_VectorField.RayCast(...)); m_pSmokeSurface->Reset(); m_bInvalidSeedLine = false;}
	bool m_bInvalidSeedLine;
};


#endif // PROGRAM_HPP_
