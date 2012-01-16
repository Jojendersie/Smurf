////////////////////////////////////////////////////////////////////////////////
//
// Smurf
// =====
// ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
//
// Smoke Surfaces: An Interactive Flow Visualization
// Technique Inspired by Real-World Flow Experiments
//
// File:              /include/glgraphics.hpp
// Author:            Martin Kirst
// Creation Date:     2011.12.30
// Description:
//
// Declaration of the OpenGL graphics engine manager.
// Provides render and viewport initialization and basic graphics functionality.
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Preprocessor Directives and Namespaces
////////////////////////////////////////////////////////////////////////////////


#ifndef GLGRAPHICS_HPP_
#define GLGRAPHICS_HPP_

#include <GL/glew.h>


////////////////////////////////////////////////////////////////////////////////
// Class Declaration
////////////////////////////////////////////////////////////////////////////////


class GLGraphics {
public:
	// Enums
	////////////////////////////////////////////////////////////////////////////////
	enum AttributeSlots {
		ASLOT_POSITION,
		ASLOT_COLOR,
		ASLOT_NORMAL,
		ASLOT_TEXCOORD0,
		ASLOT_TEXCOORD1,
		ASLOT_ID,
		ASLOT_ADJACENT,
		ASLOT_SPECIAL0,
		ASLOT_SPECIAL1,
		ASLOT_SPECIAL2
	};

	// Constructors and Destructor
	////////////////////////////////////////////////////////////////////////////////
	GLGraphics(unsigned int maxNumAttributes = 16, unsigned int maxNumTextureUnits = 48);
	~GLGraphics();

	// Accessors
	////////////////////////////////////////////////////////////////////////////////
	size_t GetMat4Size() const;
	size_t GetMat3Size() const;
	GLuint GetStandardUboIndex() const;
	unsigned int GetMaxNumAttributes() const;
	unsigned int GetMaxNumTextureUnits() const;
	GLenum GetActiveTextureStage() const;
	GLuint GetActiveShaderProgramId() const;
	GLuint GetActiveUboId() const;
	GLuint GetActiveTextureId(unsigned int textureUnit) const;
	GLuint GetActiveSamplerId(unsigned int textureUnit) const;
	void SetActiveTextureStage(GLenum textureStage);
	void SetActiveShaderProgramId(GLuint shaderProgramId);
	void SetActiveUboId(GLuint uboId);
	void SetActiveTextureId(unsigned int textureUnit, GLuint textureId);
	void SetActiveSamplerId(unsigned int textureUnit, GLuint samplerId);

	// Public Methods
	////////////////////////////////////////////////////////////////////////////////
	void InitializeGraphics();
	void ClearBuffers();
	void AdjustViewport(unsigned int width, unsigned int height);

private:
	// Variables
	////////////////////////////////////////////////////////////////////////////////
	size_t mat4Size;
	size_t mat3Size;
	GLuint standardUboIndex;
	unsigned int maxNumAttributes;
	unsigned int maxNumTextureUnits;
	GLenum activeTextureStage;
	GLuint activeShaderProgramId;
	GLuint activeUboId;
	GLuint* activeTextureIds;
	GLuint* activeSamplerIds;
};


#endif // GLGRAPHICS_HPP_
