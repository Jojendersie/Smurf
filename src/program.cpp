////////////////////////////////////////////////////////////////////////////////
//
// Smurf
// =====
// ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
//
// Smoke Surfaces: An Interactive Flow Visualization
// Technique Inspired by Real-World Flow Experiments
//
// File:              /src/program.cpp
// Author:            Martin Kirst
// Creation Date:     2011.11.16
// Description:
//
// Implementation and source code file of the main program instance.
// The main program instance is responsible for creating the window, managing the runtime behavior,
// controlling the main program sequence, reserving the render context
// and running the main execution loop.
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Preprocessor Directives and Namespaces
////////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <GL/glew.h>
#include "globals.hpp"
#include "program.hpp"


////////////////////////////////////////////////////////////////////////////////
// Definition: Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
Program::Program() {
	// set a valid video mode
	sf::VideoMode mode(Globals::RENDER_VIEWPORT_WIDTH, Globals::RENDER_VIEWPORT_HEIGHT, Globals::RENDER_COLOR_DEPTH);
	if (!mode.IsValid())
		mode = sf::VideoMode::GetDesktopMode();
	// set window style
	sf::Uint32 style = sf::Style::Close;
	if (Globals::RENDER_FULLSCREEN)
		style = style | sf::Style::Fullscreen;
	// set render context settings
	sf::ContextSettings settings(Globals::RENDER_BUFFER_DEPTH, Globals::RENDER_BUFFER_STENCIL, Globals::RENDER_ANTIALIASING_LEVEL);
	// create window with above-defined settings
	mainWindow.Create(mode, Globals::PROGRAM_TITLE, style, settings);
	// define additional window settings
	if (Globals::PROGRAM_OPEN_CENTERED) {
		sf::VideoMode desktop = sf::VideoMode::GetDesktopMode();
		mainWindow.SetPosition(desktop.Width / 2 - Globals::RENDER_VIEWPORT_WIDTH / 2, desktop.Height / 2 - Globals::RENDER_VIEWPORT_HEIGHT / 2);
	}
	mainWindow.SetFramerateLimit(Globals::RENDER_FRAMERATE_MAX);
	mainWindow.EnableVerticalSync(Globals::RENDER_VSYNC);
}


////////////////////////////////////////////////////////////////////////////////
Program::~Program() {
	delete camera;
	delete flatShader;
	delete graphics;
}


////////////////////////////////////////////////////////////////////////////////
// Definition: Accessors
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
bool Program::IsRunning() const {
	return mainWindow.IsOpened();
}


////////////////////////////////////////////////////////////////////////////////
const unsigned int& Program::GetElapsedTime() {
	return timeCurrent;
}


////////////////////////////////////////////////////////////////////////////////
const unsigned long long& Program::GetTotalTime() {
	return timeTotal;
}


////////////////////////////////////////////////////////////////////////////////
float Program::GetFramerate() {
	float weightRatio = .3f;
	float time = (1.f - weightRatio) * timeCurrent + weightRatio * timeLast;
	float fps = 1000.f / time;
	return (fps < Globals::RENDER_FRAMERATE_MAX - 1) ? fps : Globals::RENDER_FRAMERATE_MAX;
}


////////////////////////////////////////////////////////////////////////////////
// Definition: Public Methods
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void Program::Run() {
	// application main loop
	mainWindow.SetActive();
	Initialize();
	while (mainWindow.IsOpened()) {
		Update();
		Draw();
		mainWindow.Display();
	}
}


////////////////////////////////////////////////////////////////////////////////
void Program::Exit() {
	mainWindow.Close();
}


////////////////////////////////////////////////////////////////////////////////
// Definition: Private Methods
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void Program::Initialize() {
	graphics->InitializeGraphics();

	// all initial code goes here

	// initialize graphics and camera
	graphics = new GLGraphics();
	camera = new SFCamera();

	// load test shader
	flatShader = new GLShader(graphics);
	flatShader->CreateStandardUniforms(GLShader::SUSET_PROJECTION_VIEW_MODEL);
	flatShader->CreateShaderProgram("res/vfx/flat_vert.glsl", "res/vfx/flat_frag.glsl", 0, 1, 0, "inPosition");
	flatShader->CreateAdvancedUniforms(1, "solidColor");
	flatShader->Use();

	// load vector field
	m_VectorField.Load("res/data/Wing_128x64x32_T0.am");
}


////////////////////////////////////////////////////////////////////////////////
void Program::Update() {
	// handle some basic events and save times
	HandleBasicEvents();
	timeLast = timeCurrent;
	timeCurrent = mainWindow.GetFrameTime();
	timeTotal += timeCurrent;

	// all update code goes here
	camera->Update();
}


////////////////////////////////////////////////////////////////////////////////
void Program::Draw() {
	graphics->ClearBuffers();

	// all draw code goes here
	flatShader->SetStandardUniform(GLShader::SUTYPE_MATRIX4_VIEW, &(camera->GetView())[0][0]);
	flatShader->SetStandardUniform(GLShader::SUTYPE_MATRIX4_PROJECTION, &(camera->GetProjection())[0][0]);
}


////////////////////////////////////////////////////////////////////////////////
void Program::HandleBasicEvents() {
	// receive and handle the basic input events
	sf::Event event;
	while (mainWindow.PollEvent(event)) {
		// close main window after clicking the window's close button
		if (event.Type == sf::Event::Closed)
			mainWindow.Close();

		// close main window after pressing Esc
		if ((event.Type == sf::Event::KeyPressed) && (event.Key.Code == Globals::INPUT_PROGRAM_EXIT))
			mainWindow.Close();

		// adjust OpenGL viewport after window resizing
		if (event.Type == sf::Event::Resized)
			graphics->AdjustViewport(event.Size.Width, event.Size.Height);
	}
}


////////////////////////////////////////////////////////////////////////////////
// Definition: Variables
////////////////////////////////////////////////////////////////////////////////


unsigned int Program::timeCurrent = 1000;
unsigned int Program::timeLast = 1000;
unsigned long long Program::timeTotal = 0;
