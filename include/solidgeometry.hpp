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
 * Creation Date:     11.12.2011
 * Content:			  Extracting the solid geometry from an AmiraMesh
 *					  by marching cube.
 *					  Lower part: Create an octree for ray castings and marching cubes.
 *					  The octree differs between inside a solid (1) and
 *					  outside (0).
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */

#include "amloader.hpp"

#ifndef SOLIDGEOMETRY_HPP_
#define SOLIDGEOMETRY_HPP_

class SolidSurface
{
	GLuint		m_uiVAO;
	GLuint		m_uiVBO;
	GLuint		m_uiIBO;
	int			m_iNumIndices;
	int			m_iNumVertices;
public:
	// Create one surface mesh for the solid parts in a vector field.
	// Input:	_pMesh - Vector field. Assuming, that the length
	//				of a vector inside the solid is 0. Then create a
	//				surface by marching cube using the vector length as
	//				density function.
	//			_iTriangles - Upper bound of useable triangles. If the
	//				geometry needs more triangles it is cut and uncomplete.
	//				try to find a sufficent buffer size.
	SolidSurface(AmiraMesh* _pMesh, int _iTriangles);
	
	~SolidSurface();

	// Set the buffers and make the rendercall
	void Render();
};

/*
the octree idea seems to be overkilled. We do not need many ray casts and
"marching cubes" is faster than "build octree"+"optimized marching cubes"

struct OctreeNode
{
	OctreeNode* apChildren[8];
};

class SolidOctree()
{
private:
	int iSizeX, iSizeY, iSizeZ;		// number of nodes in lowest level. The octree can be irregular.
public:
}*/

#endif // SOLIDGEOMETRY_HPP_