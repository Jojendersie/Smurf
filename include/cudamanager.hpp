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
	void SetVertices(GLuint *vbo);

	void Integrate(float stepsize, unsigned int bitmask);

	void Clear();

	void RandomInit(float *a, unsigned int uiSize);
	void PrintResult(float *result, unsigned int uiSize);

	static const int	INTEGRATION_FILTER_POINT	= 0x0001;
	static const int	INTEGRATION_FILTER_LINEAR	= 0x0010;
	static const int	INTEGRATION_EULER			= 0x00010000;
	static const int	INTEGRATION_MODEULER		= 0x00100000;

private:

	unsigned int m_uiElementSize;
	unsigned int m_uiBlockSize;
	unsigned int m_uiGridSize;
	
	float *m_fDeviceVectorField;
	GLuint *vbo;
	glm::vec3 m_vSizeField;
};


#endif // CUDAMANAGER_HPP_
