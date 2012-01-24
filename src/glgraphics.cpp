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
#include "globals.hpp"
#include "glgraphics.hpp"


////////////////////////////////////////////////////////////////////////////////
// Definition: Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
GLGraphics::GLGraphics(unsigned int maxNumAttributes, unsigned int maxNumTextureUnits) {
	// set OpenGL specific settings
	mat4Size = sizeof(GLfloat) * 16;
	mat3Size = sizeof(GLfloat) * 9;
	standardUboIndex = 0;

	// initialize graphics settings
	this->maxNumAttributes = maxNumAttributes;
	if (this->maxNumAttributes == 0)
		this->maxNumAttributes = 1;
	this->maxNumTextureUnits = maxNumTextureUnits;
	activeTextureStage = GL_TEXTURE0;
	activeShaderProgramId = -1;
	activeUboId = 0;

	// allocate new memory for the texture register to save an active texture per texture unit
	activeTextureIds = 0;
	activeTextureIds = new GLuint[maxNumTextureUnits];
	if (!activeTextureIds) {
		std::cerr << "Could not reserve new memory.\n";
		std::exit(EXIT_FAILURE);
	}
	for (unsigned int i = 0; i < maxNumTextureUnits; i++)
		activeTextureIds[i] = 0;

	// allocate new memory for the sampler register to save an active sampler per texture unit
	activeSamplerIds = 0;
	activeSamplerIds = new GLuint[maxNumTextureUnits];
	if (!activeSamplerIds) {
		std::cerr << "Could not reserve new memory.\n";
		std::exit(EXIT_FAILURE);
	}
	for (unsigned int i = 0; i < maxNumTextureUnits; i++)
		activeSamplerIds[i] = 0;
}


////////////////////////////////////////////////////////////////////////////////
GLGraphics::~GLGraphics() {
	if (activeTextureIds)
		delete[] activeTextureIds;
	if (activeSamplerIds)
		delete[] activeSamplerIds;
}


////////////////////////////////////////////////////////////////////////////////
// Definition: Accessors
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
size_t GLGraphics::GetMat4Size() const {
	return mat4Size;
}


////////////////////////////////////////////////////////////////////////////////
size_t GLGraphics::GetMat3Size() const {
	return mat3Size;
}


////////////////////////////////////////////////////////////////////////////////
GLuint GLGraphics::GetStandardUboIndex() const {
	return standardUboIndex;
}


////////////////////////////////////////////////////////////////////////////////
unsigned int GLGraphics::GetMaxNumAttributes() const {
	return maxNumAttributes;
}


////////////////////////////////////////////////////////////////////////////////
unsigned int GLGraphics::GetMaxNumTextureUnits() const {
	return maxNumTextureUnits;
}


////////////////////////////////////////////////////////////////////////////////
GLenum GLGraphics::GetActiveTextureStage() const {
	return activeTextureStage;
}


////////////////////////////////////////////////////////////////////////////////
GLuint GLGraphics::GetActiveShaderProgramId() const {
	return activeShaderProgramId;
}


////////////////////////////////////////////////////////////////////////////////
GLuint GLGraphics::GetActiveUboId() const {
	return activeUboId;
}


////////////////////////////////////////////////////////////////////////////////
GLuint GLGraphics::GetActiveTextureId(unsigned int textureUnit) const {
	return (textureUnit > maxNumTextureUnits - 1) ? 0 : activeTextureIds[textureUnit];
}


////////////////////////////////////////////////////////////////////////////////
GLuint GLGraphics::GetActiveSamplerId(unsigned int textureUnit) const {
	return (textureUnit > maxNumTextureUnits - 1) ? 0 : activeSamplerIds[textureUnit];
}


////////////////////////////////////////////////////////////////////////////////
void GLGraphics::SetActiveTextureStage(GLenum textureStage) {
	activeTextureStage = textureStage;
}


////////////////////////////////////////////////////////////////////////////////
void GLGraphics::SetActiveShaderProgramId(GLuint shaderProgramId) {
	activeShaderProgramId = shaderProgramId;
}


////////////////////////////////////////////////////////////////////////////////
void GLGraphics::SetActiveUboId(GLuint uboId) {
	activeUboId = uboId;
}


////////////////////////////////////////////////////////////////////////////////
void GLGraphics::SetActiveTextureId(unsigned int textureUnit, GLuint textureId) {
	if (textureUnit > maxNumTextureUnits - 1)
		return;
	activeTextureIds[textureUnit] = textureId;
}


////////////////////////////////////////////////////////////////////////////////
void GLGraphics::SetActiveSamplerId(unsigned int textureUnit, GLuint samplerId) {
	if (textureUnit > maxNumTextureUnits - 1)
		return;
	activeSamplerIds[textureUnit] = samplerId;
}


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
	// enable Z-buffer read and write
	glEnable(GL_DEPTH_TEST);
	glDepthMask(GL_TRUE);
	glDepthFunc(GL_GREATER);

	// set color, depth and stencil buffer clear value
	glClearColor(0.f, 0.f, 0.f, 0.f);
	glClearDepth(0.f);
	glClearStencil(0);

	//filled polygons GL_LINE for wireframe
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINES);

	//enable alphablending
	//glEnable(GL_ALPHA_TEST);
	glEnable(GL_BLEND);
	//glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	//glBlendFunc(GL_ONE,GL_ONE);
}


////////////////////////////////////////////////////////////////////////////////
void GLGraphics::ClearBuffers() {
	// clear the buffers
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
}


////////////////////////////////////////////////////////////////////////////////
void GLGraphics::AdjustViewport(unsigned int width, unsigned int height) {
	// readjust the viewport
	glViewport(0, 0, width, height);
}
