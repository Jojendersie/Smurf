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
class AmiraMesh;
class SmokeSurface;

class CudaManager
{
public:
	CudaManager(AmiraMesh *_VectorField);
	~CudaManager();

	void AllocateMemory(unsigned int uiSizeVertices);

	void SetVectorField();
	void RegisterVertices(GLuint *pbo, unsigned int columns, unsigned int rows);

	void Integrate(float tInterpolate, unsigned int t0, unsigned int t1, float stepsize, unsigned int bitmask);
	void ReleaseNextColumn(SmokeSurface* _Surface);

	void Reset(SmokeSurface* _Surface);

	unsigned int GetLastReleasedColumn(){return releasedColumns;}

	unsigned int GetNumColumns() {return columns;}

	void RandomInit(float *a, unsigned int uiSize);
	void PrintResult(float *result, unsigned int uiSize);

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

	AmiraMesh* m_pVectorField;

	cudaDeviceProp cudaProp;
	static int device;
};


#endif // CUDAMANAGER_HPP_
