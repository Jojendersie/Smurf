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

float *CudaManager::m_fDeviceVectorField=NULL;
AmiraMesh *CudaManager::m_pVectorField=NULL;
cudaDeviceProp CudaManager::cudaProp;
int CudaManager::device=-1;

////////////////////////////////////////////////////////////////////////////////
// Definition: Constructors and Destructor
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
Program::Program() 
{
	timeCurrent=timeLast=1000;
	timeTotal=0;

	m_bWireframe=false;
	m_bStopProgram = false;
	m_bCloseRequest = false;
	m_bNoisyIntegration = false;
	m_bUseLinearFilter = false;
	m_bUseAdvancedEuler = true;
	m_bMouseActive = false;
	m_bUseCPUIntegration=false;
	m_uiFrameCount = 0;
	m_normalizer=0;
	m_uiEditSeedLine = 0;
	m_timeIntegrate=m_timeRender=0;

	// set a valid video mode
	sf::VideoMode mode(Globals::RENDER_VIEWPORT_WIDTH, Globals::RENDER_VIEWPORT_HEIGHT, Globals::RENDER_COLOR_DEPTH);
	if (!mode.isValid())
		mode = sf::VideoMode::getDesktopMode();
	// set window style
	sf::Uint32 style = sf::Style::Close;
	if (Globals::RENDER_FULLSCREEN)
		style = style | sf::Style::Fullscreen;
	// set render context settings
	sf::ContextSettings settings(Globals::RENDER_BUFFER_DEPTH, Globals::RENDER_BUFFER_STENCIL, Globals::RENDER_ANTIALIASING_LEVEL);
	// create window with above-defined settings
	mainWindow.create(mode, Globals::PROGRAM_TITLE, style, settings);
	// define additional window settings
	if (Globals::PROGRAM_OPEN_CENTERED) {
		sf::VideoMode desktop = sf::VideoMode::getDesktopMode();
		mainWindow.setPosition(sf::Vector2i(desktop.width / 2 - Globals::RENDER_VIEWPORT_WIDTH / 2, desktop.height / 2 - Globals::RENDER_VIEWPORT_HEIGHT / 2));
	}
	mainWindow.setFramerateLimit(Globals::RENDER_FRAMERATE_MAX);
	mainWindow.setVerticalSyncEnabled(Globals::RENDER_VSYNC);

	smokeFBO = new GLuint[RENDER_DEPTH_PEELING_LAYER];
	colorTex = new GLuint[RENDER_DEPTH_PEELING_LAYER];
	depthTex = new GLuint[RENDER_DEPTH_PEELING_LAYER];
	cudamanager = new CudaManager*[PROGRAM_NUM_SEEDLINES];
	m_pSmokeSurface = new SmokeSurface*[PROGRAM_NUM_SEEDLINES];
}


////////////////////////////////////////////////////////////////////////////////
Program::~Program() {
	delete graphics;

	delete flatShader;
	delete alphaShader;
	delete renderQuadShader;
	delete compositingShader;

	glDeleteBuffers(RENDER_DEPTH_PEELING_LAYER,smokeFBO);
	glDeleteTextures(RENDER_DEPTH_PEELING_LAYER,colorTex);
	glDeleteTextures(RENDER_DEPTH_PEELING_LAYER,depthTex);

	delete smokeFBO;
	delete colorTex;
	delete depthTex;

	delete camera;
	delete m_pSolidSurface;

	for(int i=0;i!=PROGRAM_NUM_SEEDLINES;i++)
	{
		delete cudamanager[i];
		delete m_pSmokeSurface[i];
	}

	delete[] cudamanager;
}


////////////////////////////////////////////////////////////////////////////////
// Definition: Accessors
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
bool Program::IsRunning() const {
	return mainWindow.isOpen();
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
void Program::Run(const char* _pcFile) {
	// application main loop

	mainWindow.setActive();
	Initialize(_pcFile);
	while (!m_bCloseRequest) {
		Update();
		Draw();
		mainWindow.display();
	}
	Exit();
}


////////////////////////////////////////////////////////////////////////////////
void Program::Exit() {
	mainWindow.close();
}


////////////////////////////////////////////////////////////////////////////////
// Definition: Private Methods
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
void Program::Initialize(const char* _pcFile) {
	// all initial code goes here

	// load vector field
	m_VectorField.Load(_pcFile);

	// initialize graphics and camera
	graphics = new GLGraphics();
	graphics->InitializeGraphics();

	camera = new SFCamera((m_VectorField.GetBoundingBoxMax()-m_VectorField.GetBoundingBoxMin()).length()*0.5f);
	glm::vec3 ttmp=m_VectorField.GetBoundingBoxMax()-m_VectorField.GetBoundingBoxMin();
	float tmp=sqrt(ttmp.x*ttmp.x+ttmp.y*ttmp.y+ttmp.z+ttmp.z);
	float zNear=tmp*0.03333333333f;
	camera->SetZNear(zNear);
	
	glGenTextures(1,&opaqueColor);
	glBindTexture(GL_TEXTURE_2D,opaqueColor);
	glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,Globals::RENDER_VIEWPORT_WIDTH,Globals::RENDER_VIEWPORT_HEIGHT,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);

	glGenTextures(1,&opaqueDepth);
	glBindTexture(GL_TEXTURE_2D,opaqueDepth);
	glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT24,Globals::RENDER_VIEWPORT_WIDTH,Globals::RENDER_VIEWPORT_HEIGHT,0,GL_DEPTH_COMPONENT,GL_UNSIGNED_BYTE,NULL);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);

	glGenFramebuffers(1,&opaqueFBO);
	glBindFramebuffer(GL_FRAMEBUFFER,opaqueFBO);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,opaqueColor,0);
	glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT ,GL_TEXTURE_2D,opaqueDepth,0);


	glGenTextures(RENDER_DEPTH_PEELING_LAYER,colorTex);
	glGenTextures(RENDER_DEPTH_PEELING_LAYER,depthTex);
	glGenFramebuffers(RENDER_DEPTH_PEELING_LAYER,smokeFBO);

	for(int i=0;i!=RENDER_DEPTH_PEELING_LAYER;i++)
	{
		glBindTexture(GL_TEXTURE_2D,colorTex[i]);
		glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA8,Globals::RENDER_VIEWPORT_WIDTH,Globals::RENDER_VIEWPORT_HEIGHT,0,GL_RGBA,GL_UNSIGNED_BYTE,NULL);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_CLAMP);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_CLAMP);

		glBindTexture(GL_TEXTURE_2D,depthTex[i]);
		glTexImage2D(GL_TEXTURE_2D,0,GL_DEPTH_COMPONENT24,Globals::RENDER_VIEWPORT_WIDTH,Globals::RENDER_VIEWPORT_HEIGHT,0,GL_DEPTH_COMPONENT,GL_UNSIGNED_BYTE,NULL);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);

	
		glBindFramebuffer(GL_FRAMEBUFFER,smokeFBO[i]);
		glFramebufferTexture2D(GL_FRAMEBUFFER,GL_COLOR_ATTACHMENT0,GL_TEXTURE_2D,colorTex[i],0);
		glFramebufferTexture2D(GL_FRAMEBUFFER,GL_DEPTH_ATTACHMENT ,GL_TEXTURE_2D,depthTex[i],0);
	}

	// check framebuffer status
	GLenum status = glCheckFramebufferStatus (GL_FRAMEBUFFER);
	switch (status)
	{
	case GL_FRAMEBUFFER_COMPLETE:
		std::cout << "FBO complete" << std::endl;
		break;
	case GL_FRAMEBUFFER_UNSUPPORTED:
		std::cout << "FBO configuration unsupported" << std::endl;
	default:
		std::cout << "FBO programmer error" << std::endl;
	}

	glBindFramebuffer(GL_FRAMEBUFFER,0);

	float posData[] = {-1,-1, 1,-1, -1,1, 1,1};
	float texData[] = {0,0, 1,0, 0,1, 1,1};

	GLuint posVBO,texVBO;

	glGenVertexArrays(1,&renderQuadVAO);
	glBindVertexArray(renderQuadVAO);

		glGenBuffers(1,&posVBO);
		glBindBuffer(GL_ARRAY_BUFFER,posVBO);
			glBufferData(GL_ARRAY_BUFFER,sizeof(posData),posData,GL_STATIC_DRAW);
			glVertexAttribPointer(GLGraphics::ASLOT_POSITION ,2,GL_FLOAT,GL_FALSE,0,0);
		glBindBuffer(GL_ARRAY_BUFFER,0);

		glGenBuffers(1,&texVBO);
		glBindBuffer(GL_ARRAY_BUFFER,texVBO);
			glBufferData(GL_ARRAY_BUFFER,sizeof(texData),texData,GL_STATIC_DRAW);
			glVertexAttribPointer(GLGraphics::ASLOT_TEXCOORD0 ,2,GL_FLOAT,GL_FALSE,0,0);
		glBindBuffer(GL_ARRAY_BUFFER,0);

		glEnableVertexAttribArray(GLGraphics::ASLOT_POSITION);
		glEnableVertexAttribArray(GLGraphics::ASLOT_TEXCOORD0);
	glBindVertexArray(0);

	// load test shader
	flatShader = new GLShader(graphics);
	flatShader->CreateShaderProgram("res/vfx/flat_vert.glsl", "res/vfx/flat_frag.glsl", 0, 2, GLGraphics::ASLOT_POSITION, "in_Position",GLGraphics::ASLOT_NORMAL, "in_Normal");
	flatShader->CreateAdvancedUniforms(2,"ProjectionView","eyePos");

	alphaShader = new GLShader(graphics);
	alphaShader->CreateShaderProgram("res/vfx/alphashader.vert", "res/vfx/alphashader.frag", "res/vfx/alphashader.geom",1,GLGraphics::ASLOT_POSITION,"in_Indices");
	alphaShader->CreateAdvancedUniforms(14,"b","ProjectionView","currentColumn","shapeStrength","invProjectionView","eyePos","k","columnStride","rowStride","viewPort","fragColor", "maxColumns","renderPass","areaConstant");
	alphaShader->Use();
	texLoc = glGetUniformLocation(alphaShader->GetShaderProgramm(),"adjTex");
	glUniform1i(texLoc, 0);
	if(texLoc==-1)
		std::cout << "Error: No such Texture Location" << std::endl;
	texLoc = glGetUniformLocation(alphaShader->GetShaderProgramm(),"depthTexture");
	glUniform1i(texLoc, 2);
	if(texLoc==-1)
		std::cout << "Error: No such Texture Location" << std::endl;
	texLoc = glGetUniformLocation(alphaShader->GetShaderProgramm(),"opaqueTexture");
	glUniform1i(texLoc, 3);
	if(texLoc==-1)
		std::cout << "Error: No such Texture Location" << std::endl;

	renderQuadShader = new GLShader(graphics);
	renderQuadShader->CreateShaderProgram("res/vfx/renderQuad.vert", "res/vfx/renderQuad.frag", 0,2,GLGraphics::ASLOT_POSITION,"in_Pos", GLGraphics::ASLOT_TEXCOORD0,"in_TexCoords");
	renderQuadShader->Use();
	texLoc = glGetUniformLocation(renderQuadShader->GetShaderProgramm(),"texSampler");
	glUniform1i(texLoc,1);
	if(texLoc==-1)
		std::cout << "Error: No such Texture Location" << std::endl;

	compositingShader = new GLShader(graphics);
	compositingShader->CreateShaderProgram("res/vfx/Compositing.vert", "res/vfx/Compositing.frag", 0,2,GLGraphics::ASLOT_POSITION,"in_Pos", GLGraphics::ASLOT_TEXCOORD0,"in_TexCoords");
	compositingShader->CreateAdvancedUniforms(1,"renderLayer");
	compositingShader->Use();
	for(int j=0;j!=RENDER_DEPTH_PEELING_LAYER;j++)
	{
		char shadeBuf[32];
		sprintf(shadeBuf,"texSampler%d",j);
		texLoc = glGetUniformLocation(compositingShader->GetShaderProgramm(),shadeBuf);
		glUniform1i(texLoc,4+j);
		if(texLoc==-1)
			std::cout << "Error: No such Texture Location" << std::endl;
	}
	texLoc = glGetUniformLocation(compositingShader->GetShaderProgramm(),"opaqueSampler");
		glUniform1i(texLoc,12);
		if(texLoc==-1)
			std::cout << "Error: No such Texture Location" << std::endl;
	
	m_pSolidSurface = new SolidSurface(&m_VectorField, RENDER_POLYGON_MAX);

	CudaManager::SetDevice();
	CudaManager::AllocateMemory(&m_VectorField);
	CudaManager::SetVectorField();

	for(int i=0;i<PROGRAM_NUM_SEEDLINES;++i)
	{
		m_pSmokeSurface[i] = new SmokeSurface(RENDER_SMURF_COLUMS, RENDER_SMURF_ROWS, m_VectorField.GetBoundingBoxMax(), m_VectorField.GetBoundingBoxMin());

		cudamanager[i] = new CudaManager();
		cudamanager[i]->SetSmokeSurfaceSize(m_pSmokeSurface[i]->GetNumVertices());
		GLuint tmpPBO=m_pSmokeSurface[i]->GetPBO();
		cudamanager[i]->RegisterVertices(&tmpPBO,RENDER_SMURF_COLUMS,RENDER_SMURF_ROWS);
	}

	std::cout<<"GPU-Integration"<<std::endl;
}


////////////////////////////////////////////////////////////////////////////////
void Program::Update() {
	// handle some basic events and save times
	HandleBasicEvents();
	timeLast = timeTotal;
	timeTotal = m_clock.getElapsedTime().asMilliseconds();
	timeCurrent = timeTotal-timeLast;
	

	// all update code goes here
	// Update camera unless the user will set seed points
	if (sf::Mouse::isButtonPressed(Globals::INPUT_CAM_RAY) && !m_bMouseActive) {
		m_bMouseActive = true;
		RayCast();
	} else {
		if(!sf::Mouse::isButtonPressed(Globals::INPUT_CAM_RAY)) m_bMouseActive = false;
		camera->Update(GetElapsedTime());
	}
	// Switch between seed lines
	for(unsigned int i=0;i<PROGRAM_NUM_SEEDLINES;++i)
		if(sf::Keyboard::isKeyPressed(sf::Keyboard::Key( sf::Keyboard::Num1+i)))
			m_uiEditSeedLine = i;

	m_normalizer++;
	m_timeStart=clock();

	for(int i=0; i<PROGRAM_NUM_SEEDLINES; ++i) if(!m_pSmokeSurface[i]->IsInvalide())
	{
		if(m_uiFrameCount++ % PROGRAM_FRAMES_PER_RELEASE == 0)
		{
			if(m_bUseCPUIntegration) 
				m_pSmokeSurface[i]->ReleaseNextColumn();
			else
				cudamanager[i]->ReleaseNextColumn(m_pSmokeSurface[i]);
		}
		unsigned int uiRenderFlags = (m_bUseAdvancedEuler	?Globals::INTEGRATION_MODEULER		: Globals::INTEGRATION_EULER)
								   | (m_bUseLinearFilter	?Globals::INTEGRATION_FILTER_LINEAR	: Globals::INTEGRATION_FILTER_POINT)
								   | (m_bNoisyIntegration	?Globals::INTEGRATION_NOISE			: 0);

		glm::vec4 interInfo=glm::vec4(0);
		float tInter=0;

		m_VectorField.GetSliceInterpolation(timeTotal,SMOKE_TIME_STEPSIZE,&interInfo,&tInter);

		float fNormalizedStepSize = RENDER_SMURF_STEPSIZE/m_VectorField.GetAverageVectorLength();
		if(m_bUseCPUIntegration)
			m_pSmokeSurface[i]->IntegrateCPU(&m_VectorField, fNormalizedStepSize, uiRenderFlags);
		else
			cudamanager[i]->Integrate(tInter,interInfo,fNormalizedStepSize, uiRenderFlags);

		/*if(!m_bStopProgram)
		{
			m_timeIntegrate+=clock()-m_timeStart;
			std::cout << "Time to integrate: " << double(m_timeIntegrate)/m_normalizer << "ms" <<std::endl;
		}*/
	}
}


////////////////////////////////////////////////////////////////////////////////
void Program::Draw() {
	
	m_timeStart=clock();

	//Render Opaque Objects
	glBindFramebuffer(GL_FRAMEBUFFER,opaqueFBO);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	flatShader->Use();
	flatShader->SetAdvancedUniform(GLShader::AUTYPE_MATRIX4,0,&(camera->GetProjection()*camera->GetView())[0][0]);
	flatShader->SetAdvancedUniform(GLShader::AUTYPE_VECTOR3,1,&(camera->GetPosition())[0]);
	m_pSolidSurface->Render();

	if(!Globals::RENDER_DEPTH_PEELING)
	{
		glBlendFunc(GL_ONE,GL_ONE);
		glDepthMask(GL_FALSE);
	}

	//transparent rendering
	alphaShader->Use();

	glm::vec2 viewPort = glm::vec2(Globals::RENDER_VIEWPORT_WIDTH,Globals::RENDER_VIEWPORT_HEIGHT);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_VECTOR2,9,&viewPort[0]);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_VECTOR3,10,SMOKE_COLOR);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 6,&SMOKE_DENSITY_CONSTANT_K);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 3,&SMOKE_SHAPE_CONSTANT);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_MATRIX4,4,&(camera->GetProjection()*camera->GetView())._inverse()[0][0]);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_VECTOR3,5,&camera->GetPosition()[0]);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 0,&SMOKE_CURVATURE_CONSTANT);
	alphaShader->SetAdvancedUniform(GLShader::AUTYPE_MATRIX4,1,&(camera->GetProjection()*camera->GetView())[0][0]);

	int LAYERS=0;
	if(!Globals::RENDER_DEPTH_PEELING)
		LAYERS=1;
	else
		LAYERS=RENDER_DEPTH_PEELING_LAYER;

	for(int k=0;k!=LAYERS;k++)//'Globals::RENDER_DEPTH_PEELING_LAYER' passes for the depth-peeling layers
	{
		glBindFramebuffer(GL_FRAMEBUFFER,smokeFBO[k]);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		//Render the Seedlines (Transparent Objects)
		for(int i=0;i<PROGRAM_NUM_SEEDLINES;++i) if(!m_pSmokeSurface[i]->IsInvalide())
		{
			if(m_bUseCPUIntegration)
			{
				glBindTexture(GL_TEXTURE_2D, m_pSmokeSurface[i]->GetVertexMap());
				glTexImage2D(GL_TEXTURE_2D,	// Target
					0,						// Mip-Level
					GL_RGB32F,				// Internal format
					m_pSmokeSurface[i]->GetNumRows(),// Width
					m_pSmokeSurface[i]->GetNumColumns(),// Height
					0,						// Border
					GL_RGB,					// Format
					GL_FLOAT,				// Type
					m_pSmokeSurface[i]->GetPoints());// Data
				glBindTexture(GL_TEXTURE_2D, 0);
			}
			else
			{
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER,m_pSmokeSurface[i]->GetPBO());
				glBindTexture(GL_TEXTURE_2D,m_pSmokeSurface[i]->GetVertexMap());
					glTexImage2D(GL_TEXTURE_2D,	// Target
					0,						// Mip-Level
					GL_RGB32F,				// Internal format
					m_pSmokeSurface[i]->GetNumRows(),	// Width
					m_pSmokeSurface[i]->GetNumColumns(),// Height
					0,						// Border
					GL_RGB,					// Format
					GL_FLOAT,				// Type
					NULL);					// Data
				glBindTexture(GL_TEXTURE_2D,0);
				glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);	
			}

			float fCurrentColumn;
			if(m_bUseCPUIntegration)
			{
				fCurrentColumn = float(m_uiFrameCount%PROGRAM_FRAMES_PER_RELEASE)/PROGRAM_FRAMES_PER_RELEASE;
				fCurrentColumn = float((m_pSmokeSurface[i]->GetLastReleasedColumn()%m_pSmokeSurface[i]->GetNumColumns())+fCurrentColumn);
				fCurrentColumn /= m_pSmokeSurface[i]->GetNumColumns();
				m_bStopProgram=m_pSmokeSurface[i]->GetNumColumns()<=m_pSmokeSurface[i]->GetLastReleasedColumn();
			}
			else
			{
				fCurrentColumn = float(m_uiFrameCount%PROGRAM_FRAMES_PER_RELEASE)/PROGRAM_FRAMES_PER_RELEASE;
				fCurrentColumn = float((cudamanager[i]->GetLastReleasedColumn()%cudamanager[i]->GetNumColumns())+fCurrentColumn);
				fCurrentColumn /= cudamanager[i]->GetNumColumns();
				m_bStopProgram=cudamanager[i]->GetNumColumns()<=cudamanager[i]->GetLastReleasedColumn();
			}
			float fColumnStride=1.0f/m_pSmokeSurface[i]->GetNumColumns();
			float fRowStride=1.0f/m_pSmokeSurface[i]->GetNumRows();
			float fMaxColumns = float(m_pSmokeSurface[i]->GetNumColumns());
			//float fAreaNormalisation = SMOKE_AREA_CONSTANT_NORMALIZATION * (float(m_VectorField.GetSizeX()*m_VectorField.GetSizeX() + m_VectorField.GetSizeY()*m_VectorField.GetSizeY() + m_VectorField.GetSizeZ()*m_VectorField.GetSizeZ()));
			float areaConstant=SMOKE_AREA_CONSTANT_NORMALIZATION;
			GLfloat renderPass=(GLfloat)k;

			alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 12, &renderPass);
			alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 13, &areaConstant);
			alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 2,&fCurrentColumn);
			alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 7,&fColumnStride);
			alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 8,&fRowStride);
			alphaShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR, 11, &fMaxColumns);

			glActiveTexture(GL_TEXTURE0);//activate the vertex positions from the actual smoke-surface
			glBindTexture(GL_TEXTURE_2D,m_pSmokeSurface[i]->GetVertexMap());

			glActiveTexture(GL_TEXTURE3);//activate the opaque depths
			glBindTexture(GL_TEXTURE_2D,opaqueDepth);

			glActiveTexture(GL_TEXTURE2);
			if(k!=0) glBindTexture(GL_TEXTURE_2D,depthTex[k-1]);

			m_pSmokeSurface[i]->Render();
		}
	}

	if(!Globals::RENDER_DEPTH_PEELING)
	{
		glBlendFunc(GL_ONE,GL_ZERO);
		glDepthMask(GL_TRUE);
	}

	glBindFramebuffer(GL_FRAMEBUFFER,0);
	glBindTexture(GL_TEXTURE_2D,0);

	/*if(!m_bStopProgram)
	{
		m_timeRender+=clock()-m_timeStart;
		std::cout << "Time to render: " << double(m_timeRender)/m_normalizer << "ms" << std::endl;
	}*/

	//const GLenum ErrorValue = glGetError();
	//int tmpE=0;
	//if(ErrorValue != GL_NO_ERROR) 
	//	tmpE++;
	//tmpE=GL_INVALID_VALUE&GL_INVALID_VALUE;

	//compose the peeled layer together through a render quad
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glDisable(GL_DEPTH_TEST);
	
	for(int j=0;j!=LAYERS;j++)
	{
		glActiveTexture(GL_TEXTURE4+j);
		glBindTexture(GL_TEXTURE_2D,colorTex[j]);
	}

	glActiveTexture(GL_TEXTURE12);
	glBindTexture(GL_TEXTURE_2D,opaqueColor);

	compositingShader->Use();
	GLfloat renderPass=LAYERS;
	compositingShader->SetAdvancedUniform(GLShader::AUTYPE_SCALAR,0,&renderPass);


	//glActiveTexture(GL_TEXTURE1);
	//glBindTexture(GL_TEXTURE_2D,depthTex[0]);

	//renderQuadShader->Use();

	glBindVertexArray(renderQuadVAO);
		glDrawArrays(GL_TRIANGLE_STRIP,0,4);
	glBindVertexArray(0);

	glEnable(GL_DEPTH_TEST);
}


////////////////////////////////////////////////////////////////////////////////
void Program::HandleBasicEvents() {
	// receive and handle the basic input events
	sf::Event event;
	while (mainWindow.pollEvent(event)) {
		// close main window after clicking the window's close button
		if (event.type == sf::Event::Closed)
			m_bCloseRequest = true; //mainWindow.Close();

		// close main window after pressing Esc
		if ((event.type == sf::Event::KeyPressed) && (event.key.code == Globals::INPUT_PROGRAM_EXIT))
			m_bCloseRequest = true;

		// Toggle noisy integration
		if ((event.type == sf::Event::KeyPressed) && (event.key.code == sf::Keyboard::N))
			m_bNoisyIntegration = !m_bNoisyIntegration;

		// Toggle filter
		if ((event.type == sf::Event::KeyPressed) && (event.key.code == sf::Keyboard::F))
			m_bUseLinearFilter = !m_bUseLinearFilter;

		// Toggle integration method
		if ((event.type == sf::Event::KeyPressed) && (event.key.code == sf::Keyboard::I))
			m_bUseAdvancedEuler = !m_bUseAdvancedEuler;

		// Toggle Wireframe
		if ((event.type == sf::Event::KeyPressed) && (event.key.code == sf::Keyboard::E))
		{
			m_bWireframe = !m_bWireframe;
			if(m_bWireframe)
				glPolygonMode(GL_FRONT_AND_BACK,GL_LINE);
			else
				glPolygonMode(GL_FRONT_AND_BACK,GL_FILL);
		}


		if((event.type == sf::Event::KeyPressed) && (event.key.code == sf::Keyboard::R))
		{
			m_bUseCPUIntegration = !m_bUseCPUIntegration;

			if(m_bUseCPUIntegration)
				std::cout<<"CPU-Integration"<<std::endl;
			else
				std::cout<<"GPU-Integration"<<std::endl;
		}

		// adjust OpenGL viewport after window resizing
		if (event.type == sf::Event::Resized)
			graphics->AdjustViewport(event.size.width, event.size.height);
	}
}

void Program::RayCast()
{
	// Determine the ray
	sf::Vector2i MousePos = sf::Mouse::getPosition(mainWindow);
	glm::vec3 vNear = glm::vec3(MousePos.x*2.0f/(float)mainWindow.getSize().x-1.0f,
		-MousePos.y*2.0f/(float)mainWindow.getSize().y+1.0f,
		camera->GetZNear());

	//glm::vec4 vN = (glm::inverse( camera->GetView()*camera->GetProjection() ) * glm::vec4(vNear, 1.0f));
	//vNear = camera->GetPosition() - glm::vec3(vN.x, vN.y, vN.z);
	glm::vec4 vN = (glm::inverse( camera->GetProjection()*camera->GetView() ) * glm::vec4(vNear, 1.0f));
	vNear = glm::vec3(vN.x, vN.y, vN.z)/vN.w - camera->GetPosition();

	// Search the new point
	glm::vec3 vRes = m_VectorField.RayCast(camera->GetPosition(), glm::normalize(vNear));

	// Insert
	if(!m_pSmokeSurface[m_uiEditSeedLine]->IsInvalide())	{
		m_pSmokeSurface[m_uiEditSeedLine]->SetSeedLineStart(vRes);
	} else {
		m_pSmokeSurface[m_uiEditSeedLine]->SetSeedLineEnd(vRes);
		m_pSmokeSurface[m_uiEditSeedLine]->Reset();
		cudamanager[m_uiEditSeedLine]->Reset(m_pSmokeSurface[m_uiEditSeedLine]);
	}
}


////////////////////////////////////////////////////////////////////////////////
// Definition: Variables
////////////////////////////////////////////////////////////////////////////////