/*
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Smurf
 * =====
 * ##### Martin Kirst, Johannes Jendersie, Christoph L�mmerhirt, Laura Osten #####
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

#ifndef AMLOADER_HPP_
#define AMLOADER_HPP_

// Struct can also be used for vertex formats (could be placed somewhere else, vector math?)
typedef struct float3 {
	float x,y,z;
} float3;

class AmiraMesh
{
private:
	int			m_iSizeX, m_iSizeY, m_iSizeZ;		// Resolution of the vector field in each direction
//	int			m_iVectorComponents;				// Number of dimensions in one vector (have to be 3 in this application)
	float3		m_vBBMin, m_vBBMax;					// Bounding box ("Real"-World-Size of the vector field
	float		m_fScaleX, m_fScaleY, m_fScaleZ;	// Factor m_iSize/(m_vBBMax-m_vBBMin) to translate positions to grid positions faster
	float*		m_pfBuffer;							// The data

	float3 SampleL(float x, float y, float z);		// Trilinear sampling; Coords have to be in grid space

public:
	// Just if nothing is loaded
	AmiraMesh(): m_pfBuffer(0)		{}
	// Release all buffers
	~AmiraMesh();

	// Load the mesh from file
	// Output: Success or not
	bool Load(char* _pcFileName);

	// Integrate one step over the vector field to determine new position
	// Input:	_vPosition - old position
	//			_fStepSize - the size of the integration step; smaller then m_fBBX/m_iSizeX recomended
	// Output: new position _fStepSize away from the old one.
	float3 Integrate(float3 _vPosition, float _fStepSize);

	// Getter
	float3 GetBoundingBoxMin()	{return m_vBBMin;}
	float3 GetBoundingBoxmax()	{return m_vBBMax;}
	bool IsLoaded()				{return m_pfBuffer!=0;}

	// Allow user acces to read the data (namly to store it on GPU)
	const float* GetData()	{return m_pfBuffer;}
};


#endif // AMLOADER_HPP_