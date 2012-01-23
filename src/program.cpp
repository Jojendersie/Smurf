﻿////////////////////////////////////////////////////////////////////////////////
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
	m_bCloseRequest = false;
	m_bNoisyIntegration = false;
	m_bInvalidSeedLine = false;
	m_bUseLinearFilter = false;
	m_bUseAdvancedEuler = true;
	m_bMouseActive = false;
	m_uiFrameCount = 0;
	m_normalizer=0;
	m_timeIntegrate=m_timeRender=0;
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
	delete m_pSmokeSurface;
	delete m_pSolidSurface;
	delete camera;
	delete flatShader;
	delete alphaShader;
	delete graphics;
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
	while (!m_bCloseRequest) {
		Update();
		Draw();
		mainWindow.Display();
	}
	mainWindow.Close();
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
	// all initial code goes here

	// initialize graphics and camera
	graphics = new GLGraphics();
	graphics->InitializeGraphics();

	camera = new SFCamera();
	camera->SetZNear(0.01f);

	glGenTextures(1,&colorTex);
	glBindTexture(GL_TEXTURE_2D,colorTex);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGB8,Globals::RENDER_VIEWPORT_WIDTH,Globals::RENDER_VIEWPORT_HEIGHT,0,GL_RGB,GL_FLOAT,0);

	glGenTextures(1,&depthTex);
	glBindTexture(GL_TEXTURE_2D,depthTex);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);


	glGenFramebuffers(1,&smokeFBO);
	glBindBuffer(GL_FRAMEBUFFER,smokeFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,colorTex,0);
	glBindBuffer(GL_FRAMEBUFFER,0);

	// load test shader
	flatShader = new GLShader(graphics);
	// "res/vfx/flat_geom.glsl"
	flatShader->CreateShaderProgram("res/vfx/flat_vert.glsl", "res/vfx/flat_frag.glsl", 0, 2, GLGraphics::ASLOT_POSITION, "in_Position",GLGraphics::ASLOT_NORMAL, "in_Normal");
	flatShader->CreateAdvancedUniforms(1,"ProjectionView");

	alphaShader = new GLShader(graphics);
	alphaShader->CreateShaderProgram("res/vfx/alphashader.vert", "res/vfx/alphashader.frag", "res/vfx/alphashader.geom",1,GLGraphics::ASLOT_POSITION,"in_Indices");
	alphaShader->CreateAdvancedUniforms(12,"b","ProjectionView","currentColumn","shapeStrength","invProjectionView","eyePos","k","columnStride","rowStride","viewPort","fragColor", "maxColumns");
	texLoc = glGetUniformLocation(alphaShader->GetShaderProgramm(),"adjTex");
	alphaShader->Use();
	glUniform1i(texLoc, 0);

	testShader = new GLShader(graphics);
	testShader->CreateShaderProgram("res/vfx/test.vert", "res/vfx/test.frag", 0,1,GLGraphics::ASLOT_POSITION,"in_Indices");
	testShader->CreateAdvancedUniforms(1,"ProjectionView");
	texLoc = glGetUniformLocation(testShader->GetShaderProgramm(),"adjTex");
	testShader->Use();
	glUniform1i(texLoc, 0);

	renderQuadShader = new GLShader(graphics);
	renderQuadShader->CreateShaderProgram("res/vfx/renderQuad.vert", "res/vfx/renderQuad.frag", 0,2,GLGraphics::ASLOT_POSITION,"in_Pos", GLGraphics::ASLOT_TEXCOORD0,"in_TexCoords");
	texLoc= glGetUniformLocation(renderQuadShader->GetShaderProgramm(),"texSampler");
	renderQuadShader->Use();
	glUniform1i(texLoc,0);

	// load vector field
	m_VectorField.Load("res\\data\\BubbleChamber_11x11x10_T0.am");
	m_pSmokeSurface = new SmokeSurface(Globals::RENDER_SMURF_COLUMS, Globals::RENDER_SMURF_ROWS, m_VectorField.GetBoundingBoxMax(), m_VectorField.GetBoundingBoxMin());
	m_pSolidSurface = new SolidSurface(&m_VectorField, 1000);

	cudamanager = new CudaManager(&m_VectorField);

	if(!Globals::RENDER_CPU_SMOKE)
	{
		cudamanager->AllocateMemory(m_pSmokeSurface->GetNumVertices());
		cudamanager->SetVectorField();
		GLuint tmpPBO=m_pSmokeSurface->GetPBO();
		cudamanager->RegisterVertices(&tmpPBO,Globals::RENDER_SMURF_COLUMS,Globals::RENDER_SMURF_ROWS);
	}
}


////////////////////////////////////////////////////////////////////////////////
void Program::Update() {
	// handle some basic events and save times
	HandleBasicEvents();
	timeLast = timeCurrent;
	timeCurrent = mainWindow.GetFrameTime();
	timeTotal += timeCurrent;

	// all update code goes here
	// Update camera unless the user will set need seed points
	if (sf::Keyboard::IsKeyPressed(Globals::INPUT_CAM_RAY) && sf::Mouse::IsButtonPressed(Globals::INPUT_CAM_ROTATION) && !m_bMouseActive) {
		m_bMouseActive = true;
		RayCast();
	} else {
		if(!sf::Mouse::IsButtonPressed(Globals::INPUT_CAM_ROTATION)) m_bMouseActive = false;
		camera->Update();
	}
	m_normalizer++;
	m_timeStart=clock();

	if(!m_bInvalidSeedLine)
	{
		if(m_uiFrameCount++ % Globals::PROGRAM_FRAMES_PER_RELEASE == 0)
		{
			if(Globals::RENDER_CPU_SMOKE) 
				m_pSmokeSurface->ReleaseNextColumn();
			else
				cudamanager->ReleaseNextColumn(m_pSmokeSurface);
		}
		unsigned int uiRenderFlags = (m_bUseAdvancedEuler	?Globals::INTEGRATION_MODEULER		: Globals::INTEGRATION_EULER)
								   | (m_bUseLinearFilter	?Globals::INTEGRATION_FILTER_LINEAR	: Globals::INTEGRATION_FILTER_POINT)
								   | (m_bNoisyIntegration	?Globals::INTEGRATION_NOISE			: 0);

		if(Globals::RENDER_CPU_SMOKE)
		{
			m_pSmokeSurface->IntegrateCPU(&m_VectorField, Globals::RENDER_SMURF_STEPSIZE,
				  uiRenderFlags);
		}
		else
		{
			//cudamanager.Integrate(Globals::RENDER_SMURF_STEPSIZE,CudaManager::INTEGRATION_MODEULER|CudaManager::INTEGRATION_FILTER_POINT);
			cudamanager->Integrate(0.5f,uiRenderFlags);
		}
		m_timeIntegrate+=clock()-m_timeStart;
		std::cout << "Time to integrate: " << double(m_timeIntegrate)/m_normalizer << "ms" <<std::endl;

		if(m_bCloseRequest)
			cudamanager->Clear();
	}
}


////////////////////////////////////////////////////////////////////////////////
void Program::Draw() {
	// TODO:
	// clear (black)
	// Solid mit Zwrite
	// bind Framebuffer
	// clear (black)
	// additive blending
	// ztest only (no writing)
	// render smoke
	// smokequad -> backbuffer
	//		mit blendformel lerp(Sb.rgb, Sb.a*RGB, Sb.a)

	glBindFramebuffer(GL_FRAMEBUFFER,smokeFBO);
	// all draw code goes here
	// set Camera
	m_timeStart=clock();
	if(Globals::RENDER_CPU_SMOKE)
	{
		glBindTexture(GL_TEXTURE_2D, m_pSmokeSurface->GetVertexMap());
		glTexImage2D(GL_TEXTURE_2D,	// Target
			0,						// Mip-Level
			GL_RGB32F,				// Internal format
			m_pSmokeSurface->GetNumRows(),// Width
			m_pSmokeSurface->GetNumColumns(),// Height
			0,						// Border
			GL_RGB,					// Format
			GL_FLOAT,				// Type
			m_pSmokeSurface->GetPoints());// Data
		glBindTexture(GL_TEXTURE_2D, 0);
	}
	else
	{
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER,m_pSmokeSurface->GetPBO());
		glBindTexture(GL_TEXTURE_2D,m_pSmokeSurface->GetVertexMap());
			glTexImage2D(GL_TEXTURE_2D,	// Target
			0,						// Mip-Level
			GL_RGB32F,				// Internal format
			m_pSmokeSurface->GetNumRows(),	// Width
			m_pSmokeSurface->GetNumColumns(),// Height
			0,						// Border
			GL_RGB,					// Format
			GL_FLOAT,				// Type
			NULL);					// Data
		glBindTexture(GL_TEXTURE_2D,0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);	
	}

	graphics->ClearBuffers();

	flatShader->Use();
	flatShader->SetAdvancedUniform(GLShader::AUTYPE_MATRIX4,0,&(camera->GetProjection()*camera->GetView())[0][0]);

	m_pSolidSurface->Render();

	//flatShader->UseNoShaderProgram();

	//glBindTexture(GL_TEXTURE_2D,m_pSmokeSurface->GetVertexMap());
	//float *pix = new float[60000];
	//glGetTexImage(GL_TEXTURE_2D,0,GL_RGB,GL_FLOAT,pix);
	//float tmp=pix[4*3+1];
	//glBindTexture(GL_TEXTURE_2D,0);

	if(!m_bInvalidSeedLine)
	{
		glBindTexture(GL_TEXTURE_2D,m_pSmokeSurface->GetVertexMap());
		alphaShader->Use();
		float fCurrentColumn;
		if(Globals::RENDER_CPU_SMOKE)
		{
			fCurrentColumn = float(m_uiFrameCount%Globals::PROGRAM_FRAMES_PER_RELEASE)/Globals::PROGRAM_FRAMES_PER_RELEASE;
			fCurrentColumn = float((m_pSmokeSurface->GetLastReleasedColumn()%m_pSmokeSurface->GetNumColumns())+fCurrentColumn);
			fCurrentColumn /= m_pSmokeSurface->GetNumColumns();
		} else
		{
			fCurrentColumn = float(m_uiFrameCount%Globals::PROGRAM_FRAMES_PER_RELEASE)/Globals::PROGRAM_FRAMES_PER_RELEASE;
			fCurrentColumn = float((cudamanager->GetLastReleasedColumn()%cudamanager->GetNumColumns())+fCurrentColumn);
			fCurrentColumn /= cudamanager->GetNumColumns();
		}
		float fColumnStride=1.0f/m_pSmokeSurface->GetNumColumns();
		float fRowStride=1.0f/m_pSmokeSurface->GetNumRows();
		float fMaxColumns = m_pSmokeSurface->GetNumColumns();
		glm::vec2 viewPort = glm::vec2(Globals::RENDER_VIEWPORT_WIDTH,Globals::RENDER_VIEWPORT_HEIGHT);
		alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 0,&Globals::SMOKE_CURVATURE_CONSTANT);
		alphaShader->SetAdvancedUniform(GLShader::AUTYPE_MATRIX4,1,&(camera->GetProjection()*camera->GetView())[0][0]);
		alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 2,&fCurrentColumn);
		alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 3,&Globals::SMOKE_SHAPE_CONSTANT);
		alphaShader->SetAdvancedUniform(GLShader::AUTYPE_MATRIX4,4,&(camera->GetProjection()*camera->GetView())._inverse()[0][0]);
		alphaShader->SetAdvancedUniform(GLShader::AUTYPE_VECTOR3,5,&camera->GetPosition()[0]);
		alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 6,&Globals::SMOKE_DENSITY_CONSTANT_K);
		alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 7,&fColumnStride);
		alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 8,&fRowStride);
		alphaShader->SetAdvancedUniform(GLShader::AUTYPE_VECTOR2,9,&viewPort[0]);
		alphaShader->SetAdvancedUniform(GLShader::AUTYPE_VECTOR3,10,Globals::SMOKE_COLOR);
		alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 11, &fMaxColumns);

		if(Globals::RENDER_POINTS) {
			testShader->Use();
			testShader->SetAdvancedUniform(GLShader::AUTYPE_MATRIX4,0,&(camera->GetProjection()*camera->GetView())[0][0]);
		}

		//Drawing geometry here
		m_pSmokeSurface->Render(false);
		testShader->Use();
		testShader->SetAdvancedUniform(GLShader::AUTYPE_MATRIX4,0,&(camera->GetProjection()*camera->GetView())[0][0]);
		m_pSmokeSurface->Render(true);
	
		alphaShader->UseNoShaderProgram();
		glBindTexture(GL_TEXTURE_2D,0);
	}

	m_timeRender+=clock()-m_timeStart;

	std::cout << "Time to render: " << double(m_timeRender)/m_normalizer << "ms" << std::endl;

	//const GLenum ErrorValue = glGetError();
	//int tmp=0;
	//if(ErrorValue != GL_NO_ERROR) 
	//	tmp++;
	//tmp=GL_INVALID_VALUE&GL_INVALID_VALUE;
	//glFlush();
	glBindFramebuffer(GL_FRAMEBUFFER,0);

	glBindTexture(GL_TEXTURE_2D,colorTex);
	renderQuadShader->Use();
	glClear(GL_COLOR_BUFFER_BIT);
	glBegin(GL_QUADS);
	{
		glTexCoord2f(0,0);
		glVertex3f(-1,-1,0);
		glTexCoord2f(0,1);
		glVertex3f(-1,1,0);
		glTexCoord2f(1,1);
		glVertex3f(1,1,0);
		glTexCoord2f(1,0);
		glVertex3f(1,-1,0);
	}
	glEnd();
	glBindTexture(GL_TEXTURE_2D,0);
}


////////////////////////////////////////////////////////////////////////////////
void Program::HandleBasicEvents() {
	// receive and handle the basic input events
	sf::Event event;
	while (mainWindow.PollEvent(event)) {
		// close main window after clicking the window's close button
		if (event.Type == sf::Event::Closed)
			m_bCloseRequest = true; //mainWindow.Close();

		// close main window after pressing Esc
		if ((event.Type == sf::Event::KeyPressed) && (event.Key.Code == Globals::INPUT_PROGRAM_EXIT))
			m_bCloseRequest = true;

		// Toggle noisy integration
		if ((event.Type == sf::Event::KeyPressed) && (event.Key.Code == sf::Keyboard::N))
			m_bNoisyIntegration = !m_bNoisyIntegration;

		// Toggle filter
		if ((event.Type == sf::Event::KeyPressed) && (event.Key.Code == sf::Keyboard::F))
			m_bUseLinearFilter = !m_bUseLinearFilter;

		// Toggle integration method
		if ((event.Type == sf::Event::KeyPressed) && (event.Key.Code == sf::Keyboard::I))
			m_bUseAdvancedEuler = !m_bUseAdvancedEuler;

		// adjust OpenGL viewport after window resizing
		if (event.Type == sf::Event::Resized)
			graphics->AdjustViewport(event.Size.Width, event.Size.Height);
	}
}

void Program::RayCast()
{
	// Determine the ray
	sf::Vector2i MousePos = sf::Mouse::GetPosition(mainWindow);
	glm::vec3 vNear = glm::vec3(MousePos.x*2.0f/(float)mainWindow.GetWidth()-1.0f,
		-MousePos.y*2.0f/(float)mainWindow.GetHeight()+1.0f,
		camera->GetZNear());

	//glm::vec4 vN = (glm::inverse( camera->GetView()*camera->GetProjection() ) * glm::vec4(vNear, 1.0f));
	//vNear = camera->GetPosition() - glm::vec3(vN.x, vN.y, vN.z);
	glm::vec4 vN = (glm::inverse( camera->GetProjection()*camera->GetView() ) * glm::vec4(vNear, 1.0f));
	vNear = glm::vec3(vN.x, vN.y, vN.z)/vN.w - camera->GetPosition();

	// Search the new point
	glm::vec3 vRes = m_VectorField.RayCast(camera->GetPosition(), glm::normalize(vNear));

	printf("Huhhuh\n");
	// Insert
	if(!m_bInvalidSeedLine)	{
		m_pSmokeSurface->SetSeedLineStart(vRes);
		m_bInvalidSeedLine=true;
	} else {
		m_pSmokeSurface->SetSeedLineEnd(vRes);
		if(Globals::RENDER_CPU_SMOKE)
			m_pSmokeSurface->Reset();
		cudamanager->Reset(m_pSmokeSurface);
		m_bInvalidSeedLine = false;
	}
}


////////////////////////////////////////////////////////////////////////////////
// Definition: Variables
////////////////////////////////////////////////////////////////////////////////


unsigned int Program::timeCurrent = 1000;
unsigned int Program::timeLast = 1000;
unsigned long long Program::timeTotal = 0;
