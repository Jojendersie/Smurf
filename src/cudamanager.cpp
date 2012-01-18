////////////////////////////////////////////////////////////////////////////////
//
// Smurf
// =====
// ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
//
// Smoke Surfaces: An Interactive Flow Visualization
// Technique Inspired by Real-World Flow Experiments
//
// File:              /src/cudamanager.cpp
// Author:            Christoph Lämmerhirt
// Creation Date:     2012.01.11
// Description:
//
// Declaration of the cudamanager.
// The cudamanager  is responsible for creating the cuda device, managing the cuda runtime behavior,
// allocating memory for vertices/vector field and running the kernel.
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Preprocessor Directives and Namespaces
////////////////////////////////////////////////////////////////////////////////

#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cudamanager.hpp"

extern "C" void integrateVectorFieldGPU(float* fVectorField, float3 *posptr, float *timeptr, unsigned int uiElementSize, unsigned int uiGridSize, 
										  unsigned int uiBlockSize, unsigned int iSizeFieldx, unsigned int iSizeFieldy, 
										  unsigned int iSizeFieldz, float stepsize, unsigned int bitmask);

 CudaManager::CudaManager()
{
	m_fDeviceVectorField=NULL;

	posRes=0;
	timeRes=0;

	memset(&cudaProp,0,sizeof(cudaDeviceProp));
	HandleError(cudaChooseDevice(&device,&cudaProp));
	cudaProp.major=1.0;
	cudaProp.minor=0.0;
	HandleError(cudaGLSetGLDevice(device));
}

CudaManager::~CudaManager()
{
	
}

void CudaManager::HandleError(cudaError_t cuError)
{
	if(cuError!=cudaSuccess)
	{
		printf("Error: %s \n",cudaGetErrorString(cuError));
		exit(EXIT_FAILURE);
	}
}

void CudaManager::AllocateMemory(glm::vec3 vSizeVectorField, unsigned int uiSizeVertices)
{
	m_vSizeField=vSizeVectorField;
	m_uiElementSize=uiSizeVertices;
	m_uiBlockSize=256;
	m_uiGridSize=static_cast<unsigned int>(ceil(static_cast<float>(m_uiElementSize)/static_cast<float>(m_uiBlockSize)));

	size_t size = static_cast<size_t>(m_vSizeField.x * m_vSizeField.y * m_vSizeField.z * 3 * sizeof(float));
	cudaMalloc(&m_fDeviceVectorField,size);
}

void CudaManager::SetVectorField(const float *fVectorField)
{
	size_t size = static_cast<size_t>(m_vSizeField.x * m_vSizeField.y * m_vSizeField.z * 3 * sizeof(float));

	cudaMemcpy(m_fDeviceVectorField,fVectorField,size,cudaMemcpyHostToDevice);
}

void CudaManager::RegisterVertices(GLuint vbo, GLuint timevbo)
{
	if(posRes!=NULL ||timeRes!=NULL)
	{
		HandleError(cudaGraphicsUnregisterResource(posRes));
		HandleError(cudaGraphicsUnregisterResource(timeRes));
		posRes=NULL;
		timeRes=NULL;
	}
	vboPos=vbo;
	vboTime=timevbo;

	HandleError(cudaGraphicsGLRegisterBuffer(&posRes,vboPos,cudaGraphicsMapFlagsNone));
	HandleError(cudaGraphicsGLRegisterBuffer(&timeRes,vboTime,cudaGraphicsMapFlagsNone));
}

void CudaManager::Clear()
{
	if(posRes!=NULL || timeRes!=NULL)
	{
		HandleError(cudaGraphicsUnregisterResource(posRes));
		HandleError(cudaGraphicsUnregisterResource(timeRes));
		posRes=NULL;
		timeRes=NULL;
	}

	if(m_fDeviceVectorField!=NULL)
	{
		cudaFree(m_fDeviceVectorField);
		m_fDeviceVectorField=NULL;
	}
}

void CudaManager::Integrate(float stepsize, unsigned int bitmask)
{
	GLfloat *devPosptr=NULL;
	GLfloat *devTimeptr=NULL;
	cudaGraphicsMapResources(1,&posRes);
	cudaGraphicsMapResources(1,&timeRes);

	size_t posSize=m_uiElementSize*sizeof(float)*3;
	size_t timeSize=m_uiElementSize*sizeof(float);

	cudaGraphicsResourceGetMappedPointer((void**)devPosptr,&posSize,posRes);
	cudaGraphicsResourceGetMappedPointer((void**)devTimeptr,&timeSize,timeRes);

	integrateVectorFieldGPU(m_fDeviceVectorField,(float3*)devPosptr,devTimeptr,m_uiElementSize,m_uiGridSize,m_uiBlockSize,static_cast<unsigned int>(m_vSizeField.x),static_cast<unsigned int>(m_vSizeField.y),static_cast<unsigned int>(m_vSizeField.z),stepsize,bitmask);

	cudaGraphicsUnmapResources(1,&posRes);
	cudaGraphicsUnmapResources(1,&timeRes);
}

void CudaManager::RandomInit(float *a, unsigned int uiSize)
{
	for(unsigned int i=0;i<uiSize*3;i++)
		a[i]=(rand()%m_uiElementSize)+(rand()*0.0001f);
}

void CudaManager::PrintResult(float *result, unsigned int uiSize)
{
	for(unsigned int i=0;i<uiSize*3;i++)
	{
		if(i%3==0)
			printf("%d X: %f\n",i,result[i]);
		else if(i%3==1)
			printf("%d Y: %f\n",i,result[i]);
		else
			printf("%d Z: %f\n",i,result[i]);
	}
	printf("\n");
}