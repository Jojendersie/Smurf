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

#include <time.h>
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cudamanager.hpp"

extern "C" void integrateVectorFieldGPU(float* fVectorField, float3 *posptr, unsigned int uiElementSize, unsigned int uiGridSize, 
										  unsigned int uiBlockSize, uint3 sizeField, uint3 rnd, float3 bbMin, float3 posgridOff, int resetcolumn, int rows, float stepsize, unsigned int bitmask);

extern "C" void resetOldColumn(float3* posptr, float3 bbMin, float3 bbMax, int columns, int rows, int resetColumn);

 CudaManager::CudaManager()
{
	m_fDeviceVectorField=NULL;
	posRes=0;
	releasedColumns=0;

	memset(&cudaProp,0,sizeof(cudaDeviceProp));
	HandleError(cudaChooseDevice(&device,&cudaProp));
	cudaProp.major=2.0;
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
		//exit(EXIT_FAILURE);
	}
}

void CudaManager::AllocateMemory(uint3 vSizeVectorField, unsigned int uiSizeVertices)
{
	m_vSizeField=vSizeVectorField;
	m_uiElementSize=uiSizeVertices;
	m_uiBlockSize=256;
	m_uiGridSize=static_cast<unsigned int>(ceil(static_cast<float>(m_uiElementSize)/static_cast<float>(m_uiBlockSize)));

	size_t size = static_cast<size_t>(m_vSizeField.x * m_vSizeField.y * m_vSizeField.z * 3 * sizeof(float));
	cudaMalloc(&m_fDeviceVectorField,size);
}

void CudaManager::SetVectorField(const float *VectorField, glm::vec3 bbMax, glm::vec3 bbMin)
{
	size_t size = static_cast<size_t>(m_vSizeField.x * m_vSizeField.y * m_vSizeField.z * 3 * sizeof(float));

	posGridOff.x=m_vSizeField.x/(bbMax.x-bbMin.x);
	posGridOff.y=m_vSizeField.y/(bbMax.y-bbMin.y);
	posGridOff.z=m_vSizeField.z/(bbMax.z-bbMin.z);

	this->bbMin.x=bbMin.x;
	this->bbMin.y=bbMin.y;
	this->bbMin.z=bbMin.z;

	this->bbMax.x=bbMax.x;
	this->bbMax.y=bbMax.y;
	this->bbMax.z=bbMax.z;

	cudaMemcpy(m_fDeviceVectorField,VectorField,size,cudaMemcpyHostToDevice);
}

void CudaManager::RegisterVertices(GLuint *pbo, unsigned int columns, unsigned int rows)
{
	if(posRes!=NULL)
	{
		HandleError(cudaGraphicsUnregisterResource(posRes));
		posRes=NULL;
	}
	HandleError(cudaGraphicsGLRegisterBuffer(&posRes,*pbo,cudaGraphicsMapFlagsNone));

	this->columns=columns;
	this->rows=rows;
}

void CudaManager::Clear()
{
	if(posRes!=NULL)
	{
		HandleError(cudaGraphicsUnregisterResource(posRes));
		posRes=NULL;
	}

	if(m_fDeviceVectorField!=NULL)
	{
		cudaFree(m_fDeviceVectorField);
		m_fDeviceVectorField=NULL;
	}
}

void CudaManager::ReleaseNextColumn()
{
	float *devPosptr=NULL;
	size_t posSize;

	if(releasedColumns>=columns)
	{
		int resetColumn=releasedColumns%columns;
		HandleError(cudaGraphicsMapResources(1,&posRes));
		HandleError(cudaGraphicsResourceGetMappedPointer((void**)&devPosptr,&posSize,posRes));

		resetOldColumn((float3*)devPosptr,bbMax,bbMin,columns,rows,resetColumn);

		HandleError(cudaGraphicsUnmapResources(1,&posRes));
	}

	releasedColumns++;
}

void CudaManager::Integrate(float stepsize, unsigned int bitmask)
{
	float *devPosptr=NULL;
	size_t posSize;

	uint3 rnd;
	rnd.x=time(NULL);
	rnd.y=releasedColumns;
	rnd.z=rand();

	HandleError(cudaGraphicsMapResources(1,&posRes));
	HandleError(cudaGraphicsResourceGetMappedPointer((void**)&devPosptr,&posSize,posRes));

	integrateVectorFieldGPU(m_fDeviceVectorField,(float3*)devPosptr,m_uiElementSize,m_uiGridSize,m_uiBlockSize,m_vSizeField,rnd,bbMin,posGridOff,releasedColumns,rows,stepsize,bitmask);

	HandleError(cudaGraphicsUnmapResources(1,&posRes));
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