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
#include "amloader.hpp"
#include "program.hpp"
#include "cudamanager.hpp"


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
	delete alphaShader;
	delete vertexmapShader;
	delete graphics;
	delete m_pSmokeSurface;
	delete m_pSolidSurface;
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
	flatShader->CreateShaderProgram("res/vfx/flat_vert.glsl", "res/vfx/flat_frag.glsl", 0, 1, GLGraphics::ASLOT_POSITION, "inPosition");
	flatShader->CreateAdvancedUniforms(1, "solidColor");
	flatShader->Use();

	alphaShader = new GLShader(graphics);
	alphaShader->CreateShaderProgram("res/vfx/alphashader.vert", "res/vfx/alphashader.frag", "res/vfx/alphashader.geom",4,GLGraphics::ASLOT_POSITION,"in_Pos",GLGraphics::ASLOT_NORMAL,"in_O_normal",GLGraphics::ASLOT_ADJACENT,"in_adj",GLGraphics::ASLOT_ID,"vertexID");
	alphaShader->CreateAdvancedUniforms(9,"b","ProjectionView","gridWidth","shapeStrength","invProjectionView","eyePos","k","maxTime","color");
	alphaShader->CreateTextures(1,"adjTex");

	vertexmapShader = new GLShader(graphics);
	vertexmapShader->CreateShaderProgram("res/vfx/vertexMap.vert","res/vfx/vertexMap.frag",NULL,1,GLGraphics::ASLOT_POSITION,"in_Pos",GLGraphics::ASLOT_ID,"in_vertIndex");
	vertexmapShader->CreateAdvancedUniforms(2,"gridWidth","gridHeight");

	glGenSamplers(1,&samplerID);
	glBindSampler(GL_SAMPLER_2D,samplerID);
	glSamplerParameteri(samplerID,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glSamplerParameteri(samplerID,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glSamplerParameteri(samplerID,GL_TEXTURE_WRAP_S,GL_CLAMP_TO_EDGE);
	glSamplerParameteri(samplerID,GL_TEXTURE_WRAP_T,GL_CLAMP_TO_EDGE);

	//int numColums=m_pSmokeSurface->GetNumColums();
	//int numRows=m_pSmokeSurface->GetNumRows();

	glGenTextures(1,&vertexmapID);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D,vertexmapID);
	glTexImage2D (GL_TEXTURE_2D, 0, GL_RGBA8, 1000/*static_cast<GLsizei>(numColums)*/, 1000/*static_cast<GLsizei>(numRows)*/, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);

	glGenFramebuffers(1,&vertexmapFB);
	glBindFramebuffer(GL_FRAMEBUFFER,vertexmapFB);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,vertexmapID,0);

	// load vector field
	m_VectorField.Load("res\\data\\BubbleChamber_11x11x10_T0.am");
	m_pSmokeSurface = new SmokeSurface(1000, 20, m_VectorField.GetBoundingBoxMax(), m_VectorField.GetBoundingBoxMin());
	m_pSolidSurface = new SolidSurface(&m_VectorField, 1000);

	cudamanager.AllocateMemory(glm::vec3(m_VectorField.GetSizeX(),m_VectorField.GetSizeY(),m_VectorField.GetSizeZ()),m_pSmokeSurface->GetNumVertices());
	cudamanager.SetVectorField(m_VectorField.GetData());
	cudamanager.RegisterVertices(m_pSmokeSurface->GetVBO(),m_pSmokeSurface->GetVBO());
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

	m_pSmokeSurface->ReleaseNextColumn();
	m_pSmokeSurface->IntegrateCPU(&m_VectorField, 10.01f);
}


////////////////////////////////////////////////////////////////////////////////
void Program::Draw() {

	graphics->ClearBuffers();

	// all draw code goes here
	// set Camera

	cudamanager.Integrate(0.5f,CudaManager::INTEGRATION_EULER|CudaManager::INTEGRATION_FILTER_POINT);

	const GLfloat numColums=m_pSmokeSurface->GetNumColums();
	const GLfloat numRows=m_pSmokeSurface->GetNumRows();
	vertexmapShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR,0,&numColums);
	vertexmapShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR,1,&numRows);
	vertexmapShader->Use();

	glBindFramebuffer(GL_FRAMEBUFFER,vertexmapFB);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	m_pSmokeSurface->Render();

	vertexmapShader->UseNoShaderProgram();

	glBindFramebuffer(GL_FRAMEBUFFER,0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	flatShader->SetStandardUniform(GLShader::SUTYPE_MATRIX4_VIEW, &(camera->GetView())[0][0]);
	flatShader->SetStandardUniform(GLShader::SUTYPE_MATRIX4_PROJECTION, &(camera->GetProjection())[0][0]);
	flatShader->Use();

	// render scene
	m_pSolidSurface->Render();

	alphaShader->SetTexture(GL_TEXTURE_2D,0,0,vertexmapID,samplerID);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 0,&Globals::SMOKE_CURVATURE_CONSTANT);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_MATRIX4,1,&(camera->GetProjection()*camera->GetView())[0][0]);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 2,&numColums);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 3,&Globals::SMOKE_SHAPE_CONSTANT);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_MATRIX4,4,&(camera->GetProjection()*camera->GetView())._inverse()[0][0]);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_VECTOR3,5,&camera->GetPosition()[0]);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 6,&Globals::SMOKE_DENSITY_CONSTANT_K);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 7,&Globals::SMOKE_MAX_TIME);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_VECTOR3,8,Globals:: SMOKE_COLOR);
	//alphaShader->Use();

	//Drawing geometry here
	m_pSmokeSurface->Render();

	glFlush();
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
