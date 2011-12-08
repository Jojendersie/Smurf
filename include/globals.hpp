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
// Creation Date:     07.12.2011
// Description:
//
// Header only collection of global settings, values and variables.
//
////////////////////////////////////////////////////////////////////////////////

// Preprocessor Directives and Namespaces
////////////////////////////////////////////////////////////////////////////////
#ifndef GLOBALS_HPP_
#define GLOBALS_HPP_

namespace Globals {
	// Program Settings
	////////////////////////////////////////////////////////////////////////////////
	const char PROGRAM_TITLE[] = "Smurf: Smoke Surfaces Visualization";
	const short PROGRAM_VERSION_MAJOR = 0;
	const short PROGRAM_VERSION_MINOR = 1;
	const short PROGRAM_VERSION_REVISION = 0;
	const short PROGRAM_VERSION_BUILD = 0;
	const bool PROGRAM_OPEN_CENTERED = true;

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

	// Input Settings
	////////////////////////////////////////////////////////////////////////////////
	const sf::Keyboard::Key INPUT_EXIT = sf::Keyboard::Escape;
}

#endif // GLOBALS_HPP_
