#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include "cudamanager.hpp"

//extern "C" float* integrateVectorFieldGPU(float* fVectorField, float3 *dptr, unsigned int uiElementSize, 
//										  unsigned int uiBlockSize, int iSizeFieldx, int iSizeFieldy, 
//										  int iSizeFieldz, float stepsize, unsigned int bitmask);
extern "C" float* integrateVectorFieldGPU(float* fVectorField, float* fVertices, float* fDeviceResultVertices, 
										  unsigned int uiElementSize, unsigned int uiBlockSize, int iSizeFieldx, 
										  int iSizeFieldy, int iSizeFieldz, float stepsize, unsigned int bitmask);

 CudaManager::CudaManager()
{
	m_fDeviceVertices=NULL;
	m_fDeviceVectorField=NULL;
	m_fDeviceResultVertices=NULL;

	vbo=0;
}

CudaManager::~CudaManager()
{
	
}

void CudaManager::AllocateMemory(glm::vec3 vSizeVectorField, unsigned int uiSizeVertices)
{
	m_vSizeField=vSizeVectorField;
	m_uiElementSize=uiSizeVertices;
	m_uiBlockSize=static_cast<unsigned int>(m_uiElementSize*0.25f);

	size_t size = static_cast<size_t>(m_vSizeField.x*3 * m_vSizeField.y*3 * m_vSizeField.z*3 * sizeof(float));
	cudaMalloc(&m_fDeviceVectorField,size);

	size=m_uiElementSize*3 * sizeof(float);
	cudaMalloc(&m_fDeviceVertices,size);
	cudaMalloc(&m_fDeviceResultVertices,size);
}

void CudaManager::SetVectorField(float *fVectorField)
{
	size_t size = static_cast<size_t>(m_vSizeField.x*3 * m_vSizeField.y*3 * m_vSizeField.z*3 * sizeof(float));

	cudaMemcpy(m_fDeviceVectorField,fVectorField,size,cudaMemcpyHostToDevice);
}

//void CudaManager::SetVertices(GLuint *vbo)
void CudaManager::SetVertices(float *fVertices)
{
	//if(vbo!=NULL)
	//	cudaGLUnregisterBufferObject(*vbo);
	//cudaGLRegisterBufferObject(*vbo);

	size_t size=m_uiElementSize*3 * sizeof(float);

	cudaMemcpy(m_fDeviceVertices,fVertices,size,cudaMemcpyHostToDevice);
}

void CudaManager::Clear()
{
	/*if(vbo!=NULL)
		cudaGLUnregisterBufferObject(*vbo);*/

	if(m_fDeviceVectorField!=NULL)
	{
		cudaFree(m_fDeviceVectorField);
		m_fDeviceVectorField=NULL;
	}

	if(m_fDeviceVertices!=NULL)
	{
		cudaFree(m_fDeviceVertices);
		m_fDeviceVertices=NULL;

		cudaFree(m_fDeviceResultVertices);
		m_fDeviceResultVertices=NULL;
	}
}

float* CudaManager::Integrate(float stepsize, unsigned int bitmask)
{
	//float3 *dptr;
	//cudaGLMapBufferObject((void**)&dptr,*vbo);
	//float* ptr = integrateVectorFieldGPU(m_fDeviceVectorField,dptr,m_uiElementSize,m_uiBlockSize,m_vSizeField.x,m_vSizeField.y,m_vSizeField.z,stepsize,bitmask);
	float* ptr = integrateVectorFieldGPU(m_fDeviceVectorField,m_fDeviceVertices,m_fDeviceResultVertices,m_uiElementSize,m_uiBlockSize,static_cast<int>(m_vSizeField.x),static_cast<int>(m_vSizeField.y),static_cast<int>(m_vSizeField.z),stepsize,bitmask);
	//cudaGLUnmapBufferObject(*vbo);
	return ptr;
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