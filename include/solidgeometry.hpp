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
public:
	// Create one surface cylinder at a specified seedline.
	// Input:	_pMesh - Vector field. Assuming, that the length
	//				of a vector inside the solid is 0. Then create a
	//				surface by marching cube using the vector length as
	//				density function.
	SolidSurface(AmiraMesh* _pMesh);
	
	~SolidSurface();

	// Set the buffers and make the rendercall
	void Render();
};

#endif // SOLIDGEOMETRY_HPP_