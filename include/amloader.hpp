/*
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Smurf
 * =====
 * ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
 *
 * Smoke Surfaces: An Interactive Flow Visualization
 * Technique Inspired by Real-World Flow Experiments
 *
 * Author:            Johannes Jendersie
 * Creation Date:     19.11.2011
 * Content:			  Loader for vector fields from amira mesh format.
 *					  The data is stored as uniform 3D field in one buffer.
 *					  x-fastest: To visit all grid points in the same order in which
 *					  they are in memory, one writes three nested loops over the z,y,x-axes,
 *					  where the loop over the x-axis is the innermost, and the loop
 *					  over the z-axis the outermost.
 * Source:			  http://www.mpi-inf.mpg.de/~weinkauf/notes/amiramesh.html
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */

#include <glm/glm.hpp>

#ifndef AMLOADER_HPP_
#define AMLOADER_HPP_


class AmiraMesh
{
private:
	int			m_iSizeX, m_iSizeY, m_iSizeZ;			// Resolution of the vector field in each direction
//	int			m_iVectorComponents;					// Number of dimensions in one vector (have to be 3 in this application)
	glm::vec3	m_vBBMin, m_vBBMax;						// Bounding box ("Real"-World-Size of the vector field
	float		m_fScaleX, m_fScaleY, m_fScaleZ;		// Factor m_iSize/(m_vBBMax-m_vBBMin) to translate positions to grid positions faster
	glm::vec3*	m_pvBuffer;								// The data

	glm::vec3	Sample(float x, float y, float z);		// Point sampling; Coords have to be in grid space
	glm::vec3	SampleL(float x, float y, float z);		// Trilinear sampling; Coords have to be in grid space

public:
	// Just if nothing is loaded
	AmiraMesh(): m_pvBuffer(0)		{}
	// Release all buffers
	~AmiraMesh();

	// Load the mesh from file
	// Output: Success or not
	bool Load(char* _pcFileName);

	// Types of integration methods and filters.
	// Filter and Methods can be combined arbitary.
	// INTEGRATION_FILTER_POINT | INTEGRATION_EULER is the fastest
	// INTEGRATION_FILTER_LINEAR | INTEGRATION_MODEULER has the best results.
	static const int	INTEGRATION_FILTER_POINT	= 0x0001;
	static const int	INTEGRATION_FILTER_LINEAR	= 0x0010;
	static const int	INTEGRATION_EULER			= 0x00010000;
	static const int	INTEGRATION_MODEULER		= 0x00100000;

	// Integrate one step over the vector field to determine new position
	// Input:	_vPosition - old position
	//			_fStepSize - the size of the integration step; smaller then m_fBBX/m_iSizeX recomended
	// Output: new position _fStepSize away from the old one.
	glm::vec3 Integrate(glm::vec3 _vPosition, float _fStepSize, int _iMethod);

	// Getter
	glm::vec3 GetBoundingBoxMin()	{return m_vBBMin;}
	glm::vec3 GetBoundingBoxMax()	{return m_vBBMax;}
	bool IsLoaded()					{return m_pvBuffer!=0;}
	int GetSizeX()					{return m_iSizeX;}
	int GetSizeY()					{return m_iSizeY;}
	int GetSizeZ()					{return m_iSizeZ;}

	// Allow user acces to read the data (namly to store it on GPU)
	const float* GetData()			{return (float*)m_pvBuffer;}
};


#endif // AMLOADER_HPP_