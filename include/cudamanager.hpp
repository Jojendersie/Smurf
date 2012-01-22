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

class CudaManager
{
public:
	CudaManager();
	~CudaManager();

	void AllocateMemory(uint3 vSizeVectorField, unsigned int uiSizeVertices);

	void SetVectorField(const float *VectorField, glm::vec3 bbMax, glm::vec3 bbMin);
	void RegisterVertices(GLuint *pbo, unsigned int columns, unsigned int rows);

	void Integrate(float stepsize, unsigned int bitmask);
	void ReleaseNextColumn();

	void Clear();

	unsigned int GetLastReleasedColumn(){return releasedColumns;}

	unsigned int GetNumColumns() {return columns;}

	void RandomInit(float *a, unsigned int uiSize);
	void PrintResult(float *result, unsigned int uiSize);

	static const int	INTEGRATION_FILTER_POINT	= 0x00000001;
	static const int	INTEGRATION_FILTER_LINEAR	= 0x00000010;
	static const int	INTEGRATION_RANDOM			= 0x00001000;
	static const int	INTEGRATION_EULER			= 0x00010000;
	static const int	INTEGRATION_MODEULER		= 0x00100000;

private:

	void HandleError(cudaError_t cuError);

	unsigned int releasedColumns;
	unsigned int columns;
	unsigned int rows;

	unsigned int m_uiElementSize;
	unsigned int m_uiBlockSize;
	unsigned int m_uiGridSize;
	
	cudaGraphicsResource *posRes;
	float *m_fDeviceVectorField;
	uint3 m_vSizeField;
	float3 bbMin,bbMax;
	float3 posGridOff;

	cudaDeviceProp cudaProp;
	int device;
};


#endif // CUDAMANAGER_HPP_
