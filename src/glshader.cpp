////////////////////////////////////////////////////////////////////////////////
//
// Smurf
// =====
// ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
//
// Smoke Surfaces: An Interactive Flow Visualization
// Technique Inspired by Real-World Flow Experiments
//
// File:              /src/glshader.cpp
// Author:            Martin Kirst
// Creation Date:     2011.12.30
// Description:
//
// Implementation and source code file of an OpenGL shader program.
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


#include <iostream>
#include <fstream>
#include <string>
#include <cstdarg>
#include "glshader.hpp"


////////////////////////////////////////////////////////////////////////////////
// Definition: Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
GLShader::GLShader(GLGraphics* graphics) {
	initialized = false;
	useExistingStandardUbo = false;
	standardUniformSet = SUSET_NONE;
	standardUboId = 0;
	shaderProgramId = 0;
	advancedUniformLocations = 0;
	textureLocations = 0;
	this->graphics = graphics;
}


////////////////////////////////////////////////////////////////////////////////
GLShader::~GLShader() {
	if (graphics->GetActiveShaderProgramId() == shaderProgramId)
		UseNoShaderProgram();
	if (graphics->GetActiveUboId() == standardUboId && !useExistingStandardUbo)
		UseNoUbo();
	if (initialized)
		glDeleteProgram(shaderProgramId);
	if (initialized && !useExistingStandardUbo)
		glDeleteBuffers(1, &standardUboId);
	if (advancedUniformLocations)
		delete[] advancedUniformLocations;
	if (textureLocations)
		delete[] textureLocations;
}


////////////////////////////////////////////////////////////////////////////////
// Definition: Accessors
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
bool GLShader::IsInitialized() const {
	return initialized;
}


////////////////////////////////////////////////////////////////////////////////
bool GLShader::IsUsingExistingStandardUbo() const {
	return useExistingStandardUbo;
}


////////////////////////////////////////////////////////////////////////////////
bool GLShader::IsUsed() const {
	return (graphics->GetActiveShaderProgramId() == shaderProgramId) ? true : false;
}


////////////////////////////////////////////////////////////////////////////////
unsigned int GLShader::GetStandardUniformSet() const {
	return standardUniformSet;
}


////////////////////////////////////////////////////////////////////////////////
GLuint GLShader::GetStandardUboId() const {
	return (initialized && standardUniformSet != SUSET_NONE) ? standardUboId : 0;
}


////////////////////////////////////////////////////////////////////////////////
void GLShader::SetStandardUniform(unsigned int standardUniformType, const GLfloat* data) {
	// stop if no standard uniform set was created or if 'SUSET_PROJECTION_VIEW_MODEL'
	// is set and the given data belongs to 'SUTYPE_MATRIX3_NORMAL'
	if (standardUniformSet == SUSET_NONE ||
		(standardUniformSet == SUSET_PROJECTION_VIEW_MODEL && standardUniformType == SUTYPE_MATRIX3_NORMAL))
		return;

	// bind the UBO when it is not already bound
	if (graphics->GetActiveUboId() != standardUboId) {
		glBindBuffer(GL_UNIFORM_BUFFER, standardUboId);
		graphics->SetActiveUboId(standardUboId);
	}

	// upload the specified data to GPU
	switch (standardUniformType) {
	case SUTYPE_MATRIX4_PROJECTION:
		glBufferSubData(GL_UNIFORM_BUFFER, 0, graphics->GetMat4Size(), data); break;

	case SUTYPE_MATRIX4_VIEW:
		glBufferSubData(GL_UNIFORM_BUFFER, graphics->GetMat4Size(), graphics->GetMat4Size(), data); break;

	case SUTYPE_MATRIX4_MODEL:
		glBufferSubData(GL_UNIFORM_BUFFER, graphics->GetMat4Size() * 2, graphics->GetMat4Size(), data); break;

	case SUTYPE_MATRIX3_NORMAL:
		glBufferSubData(GL_UNIFORM_BUFFER, graphics->GetMat4Size() * 3, graphics->GetMat3Size(), data); break;

	default: break;
	}
}


////////////////////////////////////////////////////////////////////////////////
void GLShader::SetAdvancedUniform(unsigned int advancedUniformType, unsigned int advancedUniformIndex, const GLfloat* data) {
	// stop if no advanced uniforms were created or if index is out of bounds
	if ((!advancedUniformLocations) ||
		(advancedUniformIndex > (sizeof(advancedUniformLocations) / sizeof(advancedUniformLocations[0])) - 1))
		return;

	// get location of the uniform by specified index
	GLint location = advancedUniformLocations[advancedUniformIndex];
	if (location < 0)
		return;

	// upload data to GPU
	switch (advancedUniformType) {
	case AUTYPE_SCALAR:
		glUniform1f(location, *data); break;

	case AUTYPE_VECTOR2:
		glUniform2fv(location, 1, data); break;

	case AUTYPE_VECTOR3:
		glUniform3fv(location, 1, data); break;

	case AUTYPE_VECTOR4:
		glUniform4fv(location, 1, data); break;

	case AUTYPE_VECTORN:
		glUniform1fv(location, sizeof(data) / sizeof(data[0]), data); break;

	case AUTYPE_MATRIX2:
		glUniformMatrix2fv(location, 1, GL_FALSE, data); break;

	case AUTYPE_MATRIX3:
		glUniformMatrix3fv(location, 1, GL_FALSE, data); break;

	case AUTYPE_MATRIX4:
		glUniformMatrix4fv(location, 1, GL_FALSE, data); break;

	default: break;
	}
}


////////////////////////////////////////////////////////////////////////////////
void GLShader::SetTexture(GLenum textureType, unsigned int textureUnit, unsigned int textureIndex, GLuint textureId, GLuint samplerId) {
	// stop if no textures were created or if index is out of bounds
	if ((!textureLocations) ||
		(textureIndex > (sizeof(textureLocations) / sizeof(textureLocations[0])) - 1))
		return;

	// get location of the texture by specified index
	GLint location = textureLocations[textureIndex];
	if (location < 0)
		return;

	// get the current texture status
	GLenum textureStage = GL_TEXTURE0 + textureUnit;
	GLuint activeTextureStage = graphics->GetActiveTextureStage();
	GLuint activeTextureId = graphics->GetActiveTextureId(textureUnit);
	GLuint activeSamplerId = graphics->GetActiveSamplerId(textureUnit);

	// stop if the texture with specified settings is already bound and continue to setting the
	// texture if needed, this procedure saves expensive unnecessary texture and sampler bindings
	if (activeTextureId == textureId && activeSamplerId == samplerId)
		return;
	if (activeTextureStage != textureStage) {
		glActiveTexture(textureStage);
		graphics->SetActiveTextureStage(textureStage);
	}
	if (activeTextureId != textureId) {
		glBindTexture(textureType, textureId);
		graphics->SetActiveTextureId(textureUnit, textureId);
	}
	if (activeSamplerId != samplerId) {
		glBindSampler(textureStage, samplerId);
		graphics->SetActiveSamplerId(textureUnit, samplerId);
	}
	glUniform1i(location, textureStage);
}


////////////////////////////////////////////////////////////////////////////////
// Definition: Public Methods
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void GLShader::CreateStandardUniforms(unsigned int standardUniformSet, GLuint existingStandardUboId) {
	// stop if shader program was already initialized
	if (initialized)
		return;

	// set the specified standard uniform set and settings
	if (standardUniformSet > SUSET_PROJECTION_VIEW_MODEL_NORMAL)
		standardUniformSet = SUSET_PROJECTION_VIEW_MODEL_NORMAL;
	this->standardUniformSet = standardUniformSet;
	if (existingStandardUboId > 0 && standardUniformSet != SUSET_NONE) {
		useExistingStandardUbo = true;
		standardUboId = existingStandardUboId;
	}
}


////////////////////////////////////////////////////////////////////////////////
void GLShader::CreateShaderProgram(const char* vertFile, const char* fragFile, const char* geomFile, unsigned int numAttributes, ...) {
	// stop if shader program was already initialized
	if (initialized)
		return;

	GLuint vertShaderId, fragShaderId, geomShaderId;

	// create shader objects
	vertShaderId = glCreateShader(GL_VERTEX_SHADER);
	fragShaderId = glCreateShader(GL_FRAGMENT_SHADER);
	if (geomFile)
		geomShaderId = glCreateShader(GL_GEOMETRY_SHADER);
	else
		geomShaderId = 0;


	// read shader source code
	if (!LoadShaderFile(vertFile, vertShaderId)) {
		std::cerr << "Shader file " << vertFile << "\ncould not be read.\n";
		glDeleteShader(vertShaderId);
		glDeleteShader(fragShaderId);
		if (geomShaderId)
			glDeleteShader(geomShaderId);
		initialized = false;
		return;
	}
	if (!LoadShaderFile(fragFile, fragShaderId)) {
		std::cerr << "Shader file " << fragFile << "\ncould not be read.\n";
		glDeleteShader(vertShaderId);
		glDeleteShader(fragShaderId);
		if (geomShaderId)
			glDeleteShader(geomShaderId);
		initialized = false;
		return;
	}
	if (geomShaderId && !LoadShaderFile(geomFile, geomShaderId)) {
		std::cerr << "Shader file " << geomFile << "\ncould not be read.\n";
		glDeleteShader(vertShaderId);
		glDeleteShader(fragShaderId);
		glDeleteShader(geomShaderId);
		initialized = false;
		return;
	}

	// compile shader objects
	glCompileShader(vertShaderId);
	glCompileShader(fragShaderId);
	if (geomShaderId)
		glCompileShader(geomShaderId);

	// check compile status of the shader objects
	if ((!CheckShaderStatus(vertShaderId, vertFile) || !CheckShaderStatus(fragShaderId, fragFile)) ||
		(geomShaderId && !CheckShaderStatus(geomShaderId, geomFile))) {
		glDeleteShader(vertShaderId);
		glDeleteShader(fragShaderId);
		if (geomShaderId)
			glDeleteShader(geomShaderId);
		initialized = false;
		return;
	}

	// create the final shader program and attach the shader objects
	shaderProgramId = glCreateProgram();
	glAttachShader(shaderProgramId, vertShaderId);
	glAttachShader(shaderProgramId, fragShaderId);
	if (geomShaderId)
		glAttachShader(shaderProgramId, geomShaderId);

	// bind the attribute names to their specific locations
	unsigned int maxNumAttributes = graphics->GetMaxNumAttributes();
	if (numAttributes > maxNumAttributes)
		numAttributes = maxNumAttributes;
	unsigned int index;
	const char* name;
	std::va_list attributes;
	va_start(attributes, numAttributes);
	for (unsigned int i = 0; i < numAttributes; i++) {
		index = va_arg(attributes, unsigned int);
		name = va_arg(attributes, const char*);
		glBindAttribLocation(shaderProgramId, index, name);
	}
	va_end(attributes);

	// attempt to link
	glLinkProgram(shaderProgramId);

	// shader objects are no longer needed
	glDeleteShader(vertShaderId);
	glDeleteShader(fragShaderId);
	if (geomShaderId)
		glDeleteShader(geomShaderId);

	// check of linking the shader program worked
	GLint testResult;
	glGetProgramiv(shaderProgramId, GL_LINK_STATUS, &testResult);
	if (testResult == GL_FALSE) {
		char infoLog[2048];
		glGetProgramInfoLog(shaderProgramId, 2048, 0, infoLog);
		std::cerr << "Shader program build up from the shader files\n"
			<< vertFile << " ,\n" << fragFile;
		if (geomShaderId)
			std::cerr << " ,\n" << geomFile;
		std::cerr << "\nfailed to link with the following error:\n"
			<< infoLog << '\n';
		glDeleteProgram(shaderProgramId);
		initialized = false;
		return;
	}

	// set true if shader program is ready to use
	initialized = true;

	// uniform set binding
	if (standardUniformSet == SUSET_NONE)
		return;
	GLuint standardUniformIndex = glGetUniformBlockIndex(shaderProgramId, "StandardUniforms");
	glUniformBlockBinding(shaderProgramId, standardUniformIndex, 0);

	// UBO creation
	if (!useExistingStandardUbo)
		CreateStandardUbo();
}


////////////////////////////////////////////////////////////////////////////////
void GLShader::CreateAdvancedUniforms(unsigned int numUniforms, ...) {
	// stop if advanced uniforms were already created or the shader program is in use
	if(advancedUniformLocations || IsUsed())
		return;

	// create array to store the locations of the uniforms from the shader
	advancedUniformLocations = new GLint[numUniforms];
	if (!advancedUniformLocations)
		return;

	// go through the function argument list to get the advanced uniforms
	const char* name;
	std::va_list uniformNames;
	va_start(uniformNames, numUniforms);
	for (unsigned int i = 0; i < numUniforms; i++) {
		name = va_arg(uniformNames, const char*);
		advancedUniformLocations[i] = glGetUniformLocation(shaderProgramId, name);
	}
	va_end(uniformNames);
}


////////////////////////////////////////////////////////////////////////////////
void GLShader::CreateTextures(unsigned int numTextures, ...) {
	// stop if textures were already created or the shader program is in use
	if (textureLocations || IsUsed())
		return;

	// check whether the specified number of textures is allowed and when not, set it to the maximum value
	unsigned int maxNumTextureUnits = graphics->GetMaxNumTextureUnits();
	if (numTextures > maxNumTextureUnits)
		numTextures = maxNumTextureUnits;

	// create array to store the locations of the texture uniforms from the shader
	textureLocations = new GLint[numTextures];
	if (!textureLocations)
		return;

	// go through the function argument list to get the textures
	const char* name;
	std::va_list textureNames;
	va_start(textureNames, numTextures);
	for (unsigned int i = 0; i < numTextures; i++) {
		name = va_arg(textureNames, const char*);
		textureLocations[i] = glGetUniformLocation(shaderProgramId, name);
	}
	va_end(textureNames);
}


////////////////////////////////////////////////////////////////////////////////
void GLShader::Use() {
	// bind the shader program when it was initialized
	if (!initialized)
		return;
	glUseProgram(shaderProgramId);
	graphics->SetActiveShaderProgramId(shaderProgramId);
}


////////////////////////////////////////////////////////////////////////////////
void GLShader::UseNoShaderProgram() {
	// unbind the current shader program if there is one bound
	if (graphics->GetActiveShaderProgramId() == 0)
		return;
	glUseProgram(0);
	graphics->SetActiveShaderProgramId(0);
}


////////////////////////////////////////////////////////////////////////////////
void GLShader::UseNoUbo() {
	// unbind the current ubo if there is one bound
	if (graphics->GetActiveUboId() == 0)
		return;
	glBindBuffer(GL_UNIFORM_BUFFER, 0);
	graphics->SetActiveUboId(0);
}


////////////////////////////////////////////////////////////////////////////////
// Definition: Private Methods
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
bool GLShader::LoadShaderFile(const char* shaderFile, GLuint shaderObjectId) {
	// open read-only file stream and continue when file is valid
	std::ifstream file(shaderFile);
	if (file.fail())
		return false;

	// read the file line by line from the stream
	std::string shaderSource;
	while (file.good()) {
		std::string line;
		std::getline(file, line);
		shaderSource.append(line + '\n');
	}

	// get the buffer from the string and set as source code
	const GLchar* shaderSourceBuffer = shaderSource.c_str();
	glShaderSource(shaderObjectId, 1, &shaderSourceBuffer, 0);

	return true;
}


////////////////////////////////////////////////////////////////////////////////
bool GLShader::CheckShaderStatus(GLuint shaderObjectId, const char* shaderFile) {
	GLint testResult;
	glGetShaderiv(shaderObjectId, GL_COMPILE_STATUS, &testResult);
	if (testResult == GL_FALSE) {
		char infoLog[2048];
		glGetShaderInfoLog(shaderObjectId, 2048, 0, infoLog);
		std::cerr << "Shader in file " << shaderFile << "\nfailed to compile with the following error:\n"
			<< infoLog << '\n';
		return false;
	}
	return true;
}


////////////////////////////////////////////////////////////////////////////////
void GLShader::CreateStandardUbo() {
	// generate buffer object
	glGenBuffers(1, &standardUboId);

	// bind buffer object as UBO
	glBindBuffer(GL_UNIFORM_BUFFER, standardUboId);

	// initialize UBO
	GLfloat identity4[] = {
		1.f, 0.f, 0.f, 0.f,
		0.f, 1.f, 0.f, 0.f,
		0.f, 0.f, 1.f, 0.f,
		0.f, 0.f, 0.f, 1.f
	};
	GLfloat identity3[] = {
		1.f, 0.f, 0.f,
		0.f, 1.f, 0.f,
		0.f, 0.f, 1.f
	};
	size_t mat4Size = graphics->GetMat4Size();
	size_t mat3Size = graphics->GetMat3Size();
	size_t uboSize = mat4Size * 3;
	if (standardUniformSet == SUSET_PROJECTION_VIEW_MODEL_NORMAL)
		uboSize += mat3Size;
	glBufferData(GL_UNIFORM_BUFFER, uboSize, 0, GL_STREAM_DRAW);

	// fill UBO with identity matrices
	glBufferSubData(GL_UNIFORM_BUFFER, 0, mat4Size, identity4);
	glBufferSubData(GL_UNIFORM_BUFFER, mat4Size, mat4Size, identity4);
	glBufferSubData(GL_UNIFORM_BUFFER, mat4Size * 2, mat4Size, identity4);
	if (standardUniformSet == SUSET_PROJECTION_VIEW_MODEL_NORMAL)
		glBufferSubData(GL_UNIFORM_BUFFER, mat4Size * 3, mat3Size, identity3);

	// bind UBO to the uniform buffer binding point specified as standard uniform buffer index
	glBindBufferRange(GL_UNIFORM_BUFFER, graphics->GetStandardUboIndex(), standardUboId, 0, uboSize);
}
