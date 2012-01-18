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

	void AllocateMemory(glm::vec3 vSizeVectorField, unsigned int uiSizeVertices);

	void SetVectorField(const float *VectorField);
	void RegisterVertices(GLuint vbo, GLuint timevbo);

	void Integrate(float stepsize, unsigned int bitmask);

	void Clear();

	void RandomInit(float *a, unsigned int uiSize);
	void PrintResult(float *result, unsigned int uiSize);

	static const int	INTEGRATION_FILTER_POINT	= 0x0001;
	static const int	INTEGRATION_FILTER_LINEAR	= 0x0010;
	static const int	INTEGRATION_EULER			= 0x00010000;
	static const int	INTEGRATION_MODEULER		= 0x00100000;

private:

	void HandleError(cudaError_t cuError);

	unsigned int m_uiElementSize;
	unsigned int m_uiBlockSize;
	unsigned int m_uiGridSize;
	
	float *m_fDeviceVectorField;
	GLuint vboPos,vboTime;
	glm::vec3 m_vSizeField;

	cudaGraphicsResource *posRes;
	cudaGraphicsResource *timeRes;

	cudaDeviceProp cudaProp;
	int device;
};


#endif // CUDAMANAGER_HPP_
