////////////////////////////////////////////////////////////////////////////////
//
// Smurf
// =====
// ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
//
// Smoke Surfaces: An Interactive Flow Visualization
// Technique Inspired by Real-World Flow Experiments
//
// File:              /src/glgraphics.cpp
// Author:            Martin Kirst
// Creation Date:     2011.12.30
// Description:
//
// Implementation and source code file of the OpenGL graphics engine manager.
// Provides render and viewport initialization and basic graphics functionality.
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Preprocessor Directives and Namespaces
////////////////////////////////////////////////////////////////////////////////


#include <iostream>
#include <cstdlib>
#include <GL/glew.h>
#include "glgraphics.hpp"


////////////////////////////////////////////////////////////////////////////////
// Definition: Public Methods
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void GLGraphics::InitializeGraphics() {
	// init GLEW
	GLenum glewInitResult = glewInit();
	if (glewInitResult != GLEW_OK) {
		std::cout << "OpenGL/GLEW Error:\n"
			<< glewGetErrorString(glewInitResult);
		std::exit(EXIT_FAILURE);
	}
	// set color, depth and stencil buffer clear value
	glClearColor(0.f, 0.f, 0.f, 0.f);
	glClearDepth(1.f);
	glClearStencil(0);
	// enable Z-buffer read and write
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
}


////////////////////////////////////////////////////////////////////////////////
void GLGraphics::ClearBuffers() {
	// clear the buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}


////////////////////////////////////////////////////////////////////////////////
void GLGraphics::AdjustViewport(const unsigned int width, const unsigned int height) {
	// readjust the viewport
	glViewport(0, 0, width, height);
}
