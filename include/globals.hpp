////////////////////////////////////////////////////////////////////////////////
//
// Smurf
// =====
// ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
//
// Smoke Surfaces: An Interactive Flow Visualization
// Technique Inspired by Real-World Flow Experiments
//
// File:              /include/globals.hpp
// Author:            Martin Kirst
// Creation Date:     2011.12.07
// Description:
//
// Header only collection of global settings, values and variables.
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Preprocessor Directives and Namespaces
////////////////////////////////////////////////////////////////////////////////


#ifndef GLOBALS_HPP_
#define GLOBALS_HPP_

#include <SFML/Window.hpp>


////////////////////////////////////////////////////////////////////////////////
// Namespace Declaration and Definiton
////////////////////////////////////////////////////////////////////////////////


namespace Globals {
	// Program Settings
	////////////////////////////////////////////////////////////////////////////////
	const char PROGRAM_TITLE[] = "Smurf: Smoke Surfaces Visualization";
	const short PROGRAM_VERSION_MAJOR = 0;
	const short PROGRAM_VERSION_MINOR = 1;
	const short PROGRAM_VERSION_REVISION = 0;
	const short PROGRAM_VERSION_BUILD = 0;
	const bool PROGRAM_OPEN_CENTERED = true;
	const unsigned short PROGRAM_FRAMES_PER_RELEASE = 1;
	const unsigned short PROGRAM_NUM_SEEDLINES = 3;

	// Render Globals
	////////////////////////////////////////////////////////////////////////////////
	const unsigned short RENDER_VIEWPORT_WIDTH = 1024;
	const unsigned short RENDER_VIEWPORT_HEIGHT = 768;
	const unsigned short RENDER_COLOR_DEPTH = 32;
	const unsigned short RENDER_BUFFER_DEPTH = 24;
	const unsigned short RENDER_BUFFER_STENCIL = 8;
	const unsigned short RENDER_ANTIALIASING_LEVEL = 4;
	const unsigned short RENDER_FRAMERATE_MAX = 120;
	const unsigned short RENDER_SMURF_COLUMS = 400;
	const unsigned short RENDER_SMURF_ROWS = 100;
	const float RENDER_SMURF_STEPSIZE = 3.75f;
	const bool RENDER_FULLSCREEN = false;
	const bool RENDER_VSYNC = true;
	const bool RENDER_POINTS = false;

	// Camera Defaults
	////////////////////////////////////////////////////////////////////////////////
	const float CAM_POSX = 0.f;
	const float CAM_POSY = 0.f;
	const float CAM_POSZ = 0.f;
	const float CAM_HEADING = 0.f;
	const float CAM_PITCH = 0.f;
	const float CAM_FOV = 45.f;
	const float CAM_ZNEAR = 0.1f;
	const float CAM_ZFAR = 1000.f;
	const float CAM_SENSITIVITY = 50.f;
	const float CAM_VELOCITY = 0.4f;

	// Input Settings
	////////////////////////////////////////////////////////////////////////////////
	const sf::Keyboard::Key INPUT_PROGRAM_EXIT = sf::Keyboard::Escape;
	const sf::Keyboard::Key INPUT_CAM_FORE = sf::Keyboard::W;
	const sf::Keyboard::Key INPUT_CAM_LEFT = sf::Keyboard::A;
	const sf::Keyboard::Key INPUT_CAM_BACK = sf::Keyboard::S;
	const sf::Keyboard::Key INPUT_CAM_RIGHT = sf::Keyboard::D;
	const sf::Mouse::Button INPUT_CAM_ROTATION = sf::Mouse::Left;
	const sf::Mouse::Button INPUT_CAM_RAY = sf::Mouse::Right;

	// Smoke Properties
	////////////////////////////////////////////////////////////////////////////////
	const unsigned int SMOKE_PARTICLE_NUMBER=10;
	const float SMOKE_PRISM_THICKNESS=0.000001f;
	const float SMOKE_DENSITY_CONSTANT=0.05f;
	const float SMOKE_SHAPE_CONSTANT=1.5f;
	const float SMOKE_CURVATURE_CONSTANT=1.0f;
	const float SMOKE_COLOR[]={0.51f,0.51f,0.51f};
	const float SMOKE_DENSITY_CONSTANT_K=SMOKE_PARTICLE_NUMBER*SMOKE_DENSITY_CONSTANT*SMOKE_PRISM_THICKNESS;

	// Types of integration methods and filters.
	////////////////////////////////////////////////////////////////////////////////
	// Filter and Methods can be combined arbitary.
	// INTEGRATION_FILTER_POINT | INTEGRATION_EULER is the fastest
	// INTEGRATION_FILTER_LINEAR | INTEGRATION_MODEULER has the best results.

	const static int	INTEGRATION_FILTER_POINT	= 0x0001;
	const static int	INTEGRATION_FILTER_LINEAR	= 0x0010;
	const static int	INTEGRATION_EULER			= 0x00010000;
	const static int	INTEGRATION_MODEULER		= 0x00100000;
	const static int	INTEGRATION_NOISE			= 0x00001000;
}


#endif // GLOBALS_HPP_
