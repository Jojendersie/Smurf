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
	const unsigned short PROGRAM_FRAMES_PER_RELEASE = 5;

	// Render Globals
	////////////////////////////////////////////////////////////////////////////////
	const unsigned short RENDER_VIEWPORT_WIDTH = 1024;
	const unsigned short RENDER_VIEWPORT_HEIGHT = 768;
	const unsigned short RENDER_COLOR_DEPTH = 32;
	const unsigned short RENDER_BUFFER_DEPTH = 24;
	const unsigned short RENDER_BUFFER_STENCIL = 8;
	const unsigned short RENDER_ANTIALIASING_LEVEL = 4;
	const unsigned short RENDER_FRAMERATE_MAX = 120;
	const bool RENDER_FULLSCREEN = false;
	const bool RENDER_VSYNC = true;

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
	const float CAM_VELOCITY = 0.3f;

	// Input Settings
	////////////////////////////////////////////////////////////////////////////////
	const sf::Keyboard::Key INPUT_PROGRAM_EXIT = sf::Keyboard::Escape;
	const sf::Keyboard::Key INPUT_CAM_FORE = sf::Keyboard::W;
	const sf::Keyboard::Key INPUT_CAM_LEFT = sf::Keyboard::A;
	const sf::Keyboard::Key INPUT_CAM_BACK = sf::Keyboard::S;
	const sf::Keyboard::Key INPUT_CAM_RIGHT = sf::Keyboard::D;
	const sf::Mouse::Button INPUT_CAM_ROTATION = sf::Mouse::Left;

	// Smoke Properties
	////////////////////////////////////////////////////////////////////////////////
	const unsigned int SMOKE_PARTICLE_NUMBER=50;
	const float SMOKE_PRISM_THICKNESS=0.2f;
	const float SMOKE_DENSTY_CONSTANT=1.0f;
	const float SMOKE_SHAPE_CONSTANT=0.75f;
	const float SMOKE_CURVATURE_CONSTANT=2.0f;
	const float SMOKE_MAX_TIME=1000.0f;
	const float SMOKE_COLOR[]={0.1f,0.1f,0.9f};
	const float SMOKE_DENSITY_CONSTANT_K=SMOKE_PARTICLE_NUMBER*SMOKE_DENSTY_CONSTANT*SMOKE_PRISM_THICKNESS;
}


#endif // GLOBALS_HPP_
