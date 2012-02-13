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


#ifndef SMOKESURFACE_HPP_
#define SMOKESURFACE_HPP_

#include "amloader.hpp"

// **************************************************************** //
// Vertex formats
struct PositionVertex {
	glm::vec3 vPosition;
};

struct GridVertex {
	float fRow;		// Row = X-Coord in vertex map
	float fColumn;
};

// **************************************************************** //
class SmokeSurface
{
	unsigned int	m_uiVAO;
	unsigned int	m_uiVBO;
	unsigned int	m_uiIBO;
	unsigned int	m_uiVertexMap;
	unsigned int	m_uiPBO;
	int				m_iNumCols;
	int				m_iNumRows;
	int				m_iNumIndices;
	int				m_iNumVertices;
	int				m_iNumReleasedColumns;		// Number of flowing columns in the vectorfield (the remaining are at the seed line), the number can be larger than the number of columns -> modolu operator (rotating cylinder)
	glm::vec3		m_vStart;
	glm::vec3		m_vEnd;
	PositionVertex*	m_pPositionMap;
	bool			m_bInvalidSeedLine;

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
	void IntegrateCPU(AmiraMesh* _pMesh, float _fStepSize, int _iMethod);

	int GetVBO();
	int GetNumColumns();
	int GetNumRows();
	int GetNumVertices();
	PositionVertex *GetPoints() {return m_pPositionMap;}
	GLuint GetVertexMap()		{return m_uiVertexMap;}
	GLuint GetPBO()				{return m_uiPBO;}
	int GetLastReleasedColumn()	{return m_iNumReleasedColumns;}
	glm::vec3 GetLineStart()	{return m_vStart;}
	glm::vec3 GetLineEnd()		{return m_vEnd;}
	bool IsInvalide()			{return m_bInvalidSeedLine;}

	// Dynamic movement of the seedline
	void SetSeedLineStart(const glm::vec3& _v)		{m_vStart = _v; m_bInvalidSeedLine=true;}
	void SetSeedLineEnd(const glm::vec3& _v)		{m_vEnd = _v; m_bInvalidSeedLine=false;}

	// Reset all vertices to the seedline
	void Reset();
};

#endif // SMOKESURFACE_HPP_