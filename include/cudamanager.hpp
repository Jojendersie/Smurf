////////////////////////////////////////////////////////////////////////////////
//
// Smurf
// =====
// ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
//
// Smoke Surfaces: An Interactive Flow Visualization
// Technique Inspired by Real-World Flow Experiments
//
// File:              /include/cudamanager.hpp
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

#ifndef CUDAMANAGER_HPP_
#define CUDAMANAGER_HPP_

#include <glm/glm.hpp>
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "amloader.hpp"

class AmiraMesh;
class SmokeSurface;

class CudaManager
{
public:
	CudaManager();
	~CudaManager();

	static void SetDevice()
	{
		m_fDeviceVectorField=NULL;
	
		if(device==-1)
		{
			//HandleError(cudaThreadExit());
			memset(&cudaProp,0,sizeof(cudaDeviceProp));
			HandleError(cudaChooseDevice(&device,&cudaProp));
			cudaProp.major=2;
			cudaProp.minor=0;

			HandleError(cudaGLSetGLDevice(device));
		}
	}

	static void AllocateMemory(AmiraMesh* _pVectorField)
	{
		m_pVectorField = _pVectorField;

		size_t size = static_cast<size_t>(m_pVectorField->GetSizeX() * m_pVectorField->GetSizeY() * m_pVectorField->GetSizeZ() * m_pVectorField->GetSizeT() * 3 * sizeof(float));
		cudaMalloc(&m_fDeviceVectorField,size);
	}

	static void SetVectorField()
	{
		size_t size = static_cast<size_t>(m_pVectorField->GetSizeX() * m_pVectorField->GetSizeY() * m_pVectorField->GetSizeZ() * m_pVectorField->GetSizeT() * 3 * sizeof(float));
		cudaMemcpy(m_fDeviceVectorField,m_pVectorField->GetData(),size,cudaMemcpyHostToDevice);
		cudaExtent cSize;
		cSize.width=m_pVectorField->GetSizeX();
		cSize.height=m_pVectorField->GetSizeY();
		cSize.depth=m_pVectorField->GetSizeZ();
		//InitCuda(m_pVectorField->GetData(),cSize);
	}

	void SetSmokeSurfaceSize(unsigned int uiSizeVertices);

	void RegisterVertices(GLuint *pbo, unsigned int columns, unsigned int rows);

	void Integrate(float tInterpolate, glm::vec4 timeSteps, float stepsize, unsigned int bitmask);
	void ReleaseNextColumn(SmokeSurface* _Surface);

	void Reset(SmokeSurface* _Surface);

	unsigned int GetLastReleasedColumn(){return releasedColumns;}

	unsigned int GetNumColumns() {return columns;}

	void RandomInit(float *a, unsigned int uiSize);
	void PrintResult(float *result, unsigned int uiSize);

private:

	static void HandleError(cudaError_t cuError)
	{
		if(cuError!=cudaSuccess)
		{
			printf("Error: %s \n",cudaGetErrorString(cuError));
			//exit(EXIT_FAILURE);
		}
	}

	unsigned int releasedColumns;
	unsigned int columns;
	unsigned int rows;

	unsigned int m_uiElementSize;
	unsigned int m_uiBlockSize;
	unsigned int m_uiGridSize;
	
	cudaGraphicsResource *posRes;

	static float *m_fDeviceVectorField;

	static AmiraMesh *m_pVectorField;

	static cudaDeviceProp cudaProp;
	static int device;
};

#endif // CUDAMANAGER_HPP_
