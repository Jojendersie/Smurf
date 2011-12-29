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

// Preprocessor Directives and Namespaces
////////////////////////////////////////////////////////////////////////////////
#include <GL/glew.h>
#include "program.hpp"
#include "globals.hpp"

// Definition: Constructors and Destructor
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

	// initialize time variables
	timeCurrent = 1000;
	timeLast = 1000;
	timeTotal = 0;
}

Program::~Program() {
}

// Definition: Accessors
////////////////////////////////////////////////////////////////////////////////
bool Program::IsRunning() const {
	return mainWindow.IsOpened();
}

unsigned int Program::GetElapsedTime() const {
	return timeCurrent;
}

unsigned long long Program::GetTotalTime() const {
	return timeTotal;
}

float Program::GetFramerate() const {
	float weightRatio = .3f;
	float time = (1.f - weightRatio) * timeCurrent + weightRatio * timeLast;
	float fps = 1000.f / time;
	return (fps < Globals::RENDER_FRAMERATE_MAX - 1) ? fps : Globals::RENDER_FRAMERATE_MAX;
}

// Definition: Public Methods
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

void Program::Exit() {
	mainWindow.Close();
}

// Definition: Private Methods
////////////////////////////////////////////////////////////////////////////////
void Program::Initialize() {
	// set color, depth and stencil buffer clear value
	glClearColor(0.f, 0.f, 0.f, 0.f);
	glClearDepth(1.f);
	glClearStencil(0);
	// enable Z-buffer read and write
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);

	// all initial code goes here

}

void Program::Update() {
	// handle some basic events and save times
	HandleBasicEvents();
	timeLast = timeCurrent;
	timeCurrent = mainWindow.GetFrameTime();
	timeTotal += timeCurrent;

	// all update code goes here

}

void Program::Draw() {
	// clear the buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);

	// all draw code goes here

}

void Program::HandleBasicEvents() {
	// receive and handle the basic input events
	sf::Event event;
	while (mainWindow.PollEvent(event)) {
		// close main window after clicking the window's close button
		if (event.Type == sf::Event::Closed)
			mainWindow.Close();

		// close main window after pressing Esc
		if ((event.Type == sf::Event::KeyPressed) && (event.Key.Code == Globals::INPUT_EXIT))
			mainWindow.Close();

		// adjust OpenGL viewport after window resizing
		if (event.Type == sf::Event::Resized)
			glViewport(0, 0, event.Size.Width, event.Size.Height);
	}
}
