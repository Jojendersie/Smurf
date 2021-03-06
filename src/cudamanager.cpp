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
#include "cudamanager.hpp"
#include "smokesurface.hpp"

extern "C" void integrateVectorFieldGPU(float* fVectorField, float3 *posptr, unsigned int uiElementSize, unsigned int uiGridSize, 
										  unsigned int uiBlockSize, uint4 sizeField, uint3 rnd, float3 bbMin, float3 posgridOff, 
										  int resetcolumn, int rows, float stepsize, unsigned int bitmask, float avgVecLength, float tInterpolate, uint4 t);

extern "C" void resetOldColumn(float3* posptr, float3 bbMin, float3 bbMax, int rows, int resetColumn);
extern "C" void InitCuda(const float *vectorField, cudaExtent size);

 CudaManager::CudaManager()
{
	posRes=0;
	releasedColumns=0;
}

CudaManager::~CudaManager()
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

void CudaManager::SetSmokeSurfaceSize(unsigned int uiSizeVertices)
{
	m_uiElementSize=uiSizeVertices;
	m_uiBlockSize=256;
	m_uiGridSize=static_cast<unsigned int>(ceil(static_cast<float>(m_uiElementSize)/static_cast<float>(m_uiBlockSize)));
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

void CudaManager::Reset(SmokeSurface* _Surface)
{
	releasedColumns = columns;
	for(unsigned int i=0; i<columns; ++i)
		ReleaseNextColumn(_Surface);
}

void CudaManager::ReleaseNextColumn(SmokeSurface* _Surface)
{
	float *devPosptr=NULL;
	size_t posSize;

	if(releasedColumns>=columns)
	{
		int resetColumn=releasedColumns%columns;
		HandleError(cudaGraphicsMapResources(1,&posRes));
		HandleError(cudaGraphicsResourceGetMappedPointer((void**)&devPosptr,&posSize,posRes));

		resetOldColumn((float3*)devPosptr,*(float3*)&_Surface->GetLineStart(),*(float3*)&_Surface->GetLineEnd(),rows,resetColumn);

		HandleError(cudaGraphicsUnmapResources(1,&posRes));
	}

	releasedColumns++;
}

void CudaManager::Integrate(float tInterpolate, glm::vec4 timeSteps, float stepsize, unsigned int bitmask)
{
	float *devPosptr=NULL;
	size_t posSize;

	uint3 rnd;
	rnd.x=(unsigned int)time(NULL);
	rnd.y=releasedColumns;
	rnd.z=rand();

	HandleError(cudaGraphicsMapResources(1,&posRes));
	HandleError(cudaGraphicsResourceGetMappedPointer((void**)&devPosptr,&posSize,posRes));

	uint4 vSizeField;
	vSizeField.x = m_pVectorField->GetSizeX();
	vSizeField.y = m_pVectorField->GetSizeY();
	vSizeField.z = m_pVectorField->GetSizeZ();
	vSizeField.w = m_pVectorField->GetSizeT();

	integrateVectorFieldGPU(m_fDeviceVectorField,(float3*)devPosptr,m_uiElementSize,m_uiGridSize,m_uiBlockSize,vSizeField,rnd,*(float3*)&m_pVectorField->GetBoundingBoxMin(),
							*(float3*)&m_pVectorField->GetPosToGridVector(),releasedColumns,rows,stepsize,bitmask,m_pVectorField->GetAverageVectorLength()*50.0f,tInterpolate,make_uint4(timeSteps.x,timeSteps.y,timeSteps.z,timeSteps.w));

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