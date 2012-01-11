////////////////////////////////////////////////////////////////////////////////
//
// Smurf
// =====
// ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
//
// Smoke Surfaces: An Interactive Flow Visualization
// Technique Inspired by Real-World Flow Experiments
//
// File:              /include/glshader.hpp
// Author:            Martin Kirst
// Creation Date:     2011.12.30
// Description:
//
// Declaration of an OpenGL shader program.
// Loads, compiles and organizes a shader program.
//
// The GLShader class can be used with the following steps:
// 1. create an GLShader instance
// 2. create a standard uniform set
// 3. create the shader program with an user defined number of attributes
// 4. create advanced uniforms
// 5. create textures
// 6. use the shader program
// 7. set the specified uniforms and textures.
//
// Standard uniforms e.g. matrices are stored in an UBO when 'standardUniformSet' is not 'SUSET_NONE'
// i.e. standard uniform variables in a shader have to be placed in an uniform block.
// When the argument 'existingStandardUboId' from the method 'CreateStandardUniforms()' is specified
// greater than 0, standard uniforms can be shared across different shader programs.
//
// Using standard uniform set 'SUSET_PROJECTION_VIEW_MODEL':
// layout(std140) uniform StandardUniforms {
//     mat4 projection;
//     mat4 view;
//     mat4 model;
// }
//
// Using standard uniform set 'SUSET_PROJECTION_VIEW_MODEL_NORMAL':
// layout(std140) uniform StandardUniforms {
//     mat4 projection;
//     mat4 view;
//     mat4 model;
//     mat3 normal;
// }
//
// The method 'CreateShaderProgram()' creates a new shader program with a definable number of
// attributes e.g. to use a position and a color attribute from a specific VAO use the method like this:
// 'CreateShaderProgram("shader_vert.glsl", "shader_frag.glsl", 0, 2, 0, "in_Position", 1, "in_Color")'.
// The unsigned int value before an attribute name string defines the location index constant of the
// attribute set in the bound VAO. OpenGL Core Profile 3.3 supports no more than 16 attributes per vertex.
//
// The 'advancedUniformIndex' is read from the order of the string arguments passed through the
// 'CreateAdvancedUniforms(unsigned int numUniforms, ...)' method as function argument list (...).
// When used 'CreateAdvancedUniforms(2, "material", "color")', the uniform variable 'material'
// is accessible e.g. with 'SetAdvancedUniform(AUTYPE_VECTOR4, 0, materialVec)' and
// 'color' e.g. with 'SetAdvancedUniform(AUTYPE_VECTOR4, 1, colorVec)'.
//
// Setting textures works the same way like setting and creating advanced uniform variables.
// OpenGL Core Profile 3.3 provides at least 48 texture units per one draw call, so the max value
// for the 'numTextures' argument of the method 'CreateTextures(unsigned int numTextures, ...)' is 48
// by default to be compatible to older graphics cards, i.e. the method takes a maximum amount of 48
// different texture name strings for the shader to setup by default.
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Preprocessor Directives and Namespaces
////////////////////////////////////////////////////////////////////////////////


#ifndef GLSHADER_HPP_
#define GLSHADER_HPP_


////////////////////////////////////////////////////////////////////////////////
// Class Declaration
////////////////////////////////////////////////////////////////////////////////


class GLShader {
public:
	// Enums
	////////////////////////////////////////////////////////////////////////////////
	enum StandardUniformSets {
		SUSET_NONE,
		SUSET_PROJECTION_VIEW_MODEL,
		SUSET_PROJECTION_VIEW_MODEL_NORMAL
	};

	enum StandardUniformTypes {
		SUTYPE_MATRIX4_PROJECTION,
		SUTYPE_MATRIX4_VIEW,
		SUTYPE_MATRIX4_MODEL,
		SUTYPE_MATRIX3_NORMAL
	};

	enum AdvancedUniformTypes {
		AUTYPE_SCALAR,
		AUTYPE_VECTOR2,
		AUTYPE_VECTOR3,
		AUTYPE_VECTOR4,
		AUTYPE_VECTORN,
		AUTYPE_MATRIX2,
		AUTYPE_MATRIX3,
		AUTYPE_MATRIX4
	};

	// Constructors and Destructor
	////////////////////////////////////////////////////////////////////////////////
	GLShader(GLGraphics* graphics);
	~GLShader();

	// Accessors
	////////////////////////////////////////////////////////////////////////////////
	bool IsInitialized() const;
	bool IsUsingExistingStandardUbo() const;
	bool IsUsed() const;
	unsigned int GetStandardUniformSet() const;
	GLuint GetStandardUboId() const;
	void SetStandardUniform(unsigned int standardUniformType, GLfloat* data);
	void SetAdvancedUniform(unsigned int advancedUniformType, unsigned int advancedUniformIndex, GLfloat* data);
	void SetTexture(GLenum textureType, unsigned int textureUnit, unsigned int textureIndex, GLuint textureId, GLuint samplerId);

	// Public Methods
	////////////////////////////////////////////////////////////////////////////////
	void CreateStandardUniforms(unsigned int standardUniformSet, GLuint existingStandardUboId = 0);
	void CreateShaderProgram(const char* vertFile, const char* fragFile, const char* geomFile, unsigned int numAttributes, ...);
	void CreateAdvancedUniforms(unsigned int numUniforms, ...);
	void CreateTextures(unsigned int numTextures, ...);
	void Use();
	void UseNoShaderProgram();
	void UseNoUbo();

private:
	// Private Methods
	////////////////////////////////////////////////////////////////////////////////
	bool LoadShaderFile(const char* shaderFile, GLuint shaderObjectId);
	bool CheckShaderStatus(GLuint shaderObjectId, const char* shaderFile);
	void CreateStandardUbo();
	GLShader(const GLShader&);
	GLShader& operator=(const GLShader&);

	// Variables
	////////////////////////////////////////////////////////////////////////////////
	bool initialized;
	bool useExistingStandardUbo;
	unsigned int standardUniformSet;
	GLuint standardUboId;
	GLuint shaderProgramId;
	GLint* advancedUniformLocations;
	GLint* textureLocations;
	GLGraphics* graphics;
};


#endif // GLSHADER_HPP_
