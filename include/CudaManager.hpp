#include <glm/glm.hpp>
#include <GL\glew.h>
#include <cuda_gl_interop.h>

class CudaManager
{
public:
	CudaManager();
	~CudaManager();

	void AllocateMemory(glm::vec3 vSizeVectorField, unsigned int uiSizeVertices);

	void SetVectorField(float *VectorField);
	void SetVertices(float *Vertices);

	float* Integrate(float stepsize, unsigned int bitmask);

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
	
	float *m_fDeviceVectorField;
	float *m_fDeviceVertices,*m_fDeviceResultVertices;
	GLuint *vbo;
	glm::vec3 m_vSizeField;
};