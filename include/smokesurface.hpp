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
 * Content:			  Create and modify the smoke surface geometry (VBOs).
 *					  It is possible to create multiple indipendent surfaces in
 *					  the fixed topology of a cylinder.
 *					  Definition: A column of the mesh are vertices along the main
 *					  axis of the cylinder.
 *					  A row is one vertex ring in the cylinder.
 *					  The vertexbuffer contains column after column (vertex order)
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 */

#include "amloader.hpp"

#ifndef SMOKESURFACE_HPP_
#define SMOKESURFACE_HPP_

class SmokeSurface
{
	unsigned int	m_uiVAO;
	unsigned int	m_uiVBO;
	unsigned int	m_uiIBO;
	int				m_iNumCols;
	int				m_iNumRows;
	int				m_iNumIndices;
	int				m_iNumReleasedColumns;		// Number of flowing columns in the vectorfield (the remaining are at the seed line), the number can be larger than the number of columns -> modolu operator (rotating cylinder)
	glm::vec3		m_vStart;
	glm::vec3		m_vEnd;
public:
	// Create one surface cylinder at a specified seedline.
	// Input:	_iNumCols, _iNumRows - detail of the surface
	//			_vStart, _vEnd - start and end point of a seed line, which
	//					later will be the line, where the columns are
	//					released.
	SmokeSurface(int _iNumCols, int _iNumRows, glm::vec3 _vStart, glm::vec3 _vEnd);
	
	~SmokeSurface();

	// Set the buffers and make the rendercall for the geometry
	void Render();

	// Sets the next column to the seed line and increase the number of columns
	void ReleaseNextColumn();

	// Input:	_pMesh - vector field that should be visualized
	//			_fStepSize - The size (distance) of the one integration step.
	//				To integrate over larger timeslices call IntegrateCPU
	//				multiple times.
	void IntegrateCPU(AmiraMesh* _pMesh, float _fStepSize);

	// IntegrateGPU(AmiraMesh* _pMesh)
};

#endif // SMOKESURFACE_HPP_