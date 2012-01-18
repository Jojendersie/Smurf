#include <cassert>
#include <GL/glew.h>
#include <cstdlib>
#include "smokesurface.hpp"
#include "glgraphics.hpp"

// **************************************************************** //
// Create one surface cylinder at a specified seedline.
SmokeSurface::SmokeSurface(int _iNumCols, int _iNumRows, glm::vec3 _vStart, glm::vec3 _vEnd)
{
	// Copy parameters
	m_iNumCols = _iNumCols;
	m_iNumRows = _iNumRows;
	m_vStart = _vStart;
	m_vEnd = _vEnd;
	m_iNumReleasedColumns = 0;

	// Create data
	m_iNumVertices = _iNumCols*_iNumRows;
	int iVertexDataSize = sizeof(PositionVertex)*m_iNumVertices;
	m_pPositionMap = (PositionVertex*)malloc(iVertexDataSize);
	GridVertex* pGridVertices = (GridVertex*)malloc(sizeof(GridVertex)*m_iNumVertices);
	for(int i=0; i<_iNumCols; ++i)
		for(int j=0; j<_iNumRows; ++j)
		{
			m_pPositionMap[i*_iNumRows+j].vPosition = glm::mix(_vStart, _vEnd, j/(float)(_iNumRows-1));
			pGridVertices[i*_iNumRows+j].fColumn = i/(_iNumCols);
			pGridVertices[i*_iNumRows+j].fRow = j/(_iNumRows);
		}

	// Create Triangulation
	m_iNumIndices = _iNumCols*(_iNumRows-1)*6;
	GLuint* pIndices = (GLuint*)malloc(m_iNumIndices*sizeof(GLuint));
	GLuint* pI = pIndices;
	for(int i=0; i<_iNumCols; ++i)
		for(int j=0; j<_iNumRows-1; ++j)
		{
			int iVertex = (i*_iNumRows+j);
			// Adding a quad (2 triangles)
			*(pI++) = iVertex;
			*(pI++) = iVertex+1;
			*(pI++) = (iVertex+_iNumRows)%m_iNumVertices;
			*(pI++) = iVertex+1;
			*(pI++) = (iVertex+_iNumRows)%m_iNumVertices;
			*(pI++) = (iVertex+_iNumRows+1)%m_iNumVertices;
		}

	// Create OpenGL buffers
	// Create Vertex array object
	glGenVertexArrays(1, &m_uiVAO);
	// Create Vertex buffer object
	glBindVertexArray(m_uiVAO);
	glGenBuffers(1, &m_uiVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_uiVBO);
	// Insert data and usage declaration
	glBufferData(GL_ARRAY_BUFFER, iVertexDataSize, pGridVertices, GL_DYNAMIC_DRAW);
	glVertexAttribPointer(GLGraphics::ASLOT_SPECIAL0, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(0);
	// Insert triangulation
	glGenBuffers(1, &m_uiIBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_uiIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_iNumIndices*sizeof(GLuint), pIndices, GL_STATIC_DRAW);

	// data is no loaded to GPU
//	free(pVertices);	//? dynamic buffers? TODO benchmarktest with just uploading and or double vbo,s
	free(pIndices);

	// Vertex Map
	// Create a texture buffer object
	glGenTextures(1, &m_uiVertexMap);
}
	
// **************************************************************** //
SmokeSurface::~SmokeSurface()
{
	// Detach and delete Vertex buffer
	glBindVertexArray(m_uiVAO);
	glDisableVertexAttribArray(0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &m_uiVBO);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
	glDeleteBuffers(1, &m_uiIBO);

	// Detach and delete array
	glBindVertexArray(0);
	glDeleteVertexArrays(1, &m_uiVAO);

	free(m_pPositionMap);
}

// **************************************************************** //
// Set the buffers and make the rendercall for the geometry
void SmokeSurface::Render()
{
	glBindVertexArray(m_uiVAO);
	glDrawArrays(GL_POINTS, 0, m_iNumVertices);
	//glDrawElements(GL_TRIANGLES, m_iNumIndices, GL_UNSIGNED_INT, (GLvoid*)0);
}

// **************************************************************** //
// Sets the next column to the seed line and increase the number of columns
void SmokeSurface::ReleaseNextColumn()
{
	// Set a base line (if in flow do nothing otherwise).
	if(m_iNumReleasedColumns >= m_iNumCols)
	{
		// Lock dynamic buffer
	/*	glBindBuffer(GL_ARRAY_BUFFER, m_uiVBO);
		PositionVertex* pVertices = (PositionVertex*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
		assert(pVertices);	// TODO real error output?*/

		int i=m_iNumReleasedColumns%m_iNumCols;
		for(int j=0; j<m_iNumRows; ++j)
			m_pPositionMap[i*m_iNumRows+j].vPosition = glm::mix(m_vStart, m_vEnd, j/(float)(m_iNumRows-1));

		// Unlock
		//glUnmapBuffer(GL_ARRAY_BUFFER);
	}
	++m_iNumReleasedColumns;
}

// **************************************************************** //
// Input:	_pMesh - vector field that should be visualized
//			_fStepSize - The size (distance) of the one integration step.
//				To integrate over larger timeslices call IntegrateCPU
//				multiple times.
void SmokeSurface::IntegrateCPU(AmiraMesh* _pMesh, float _fStepSize)
{
	// Integrate everything in flow (streak surface)
	int iNumCols = glm::min(m_iNumCols, m_iNumReleasedColumns);

	// Lock dynamic buffer
	//glBindBuffer(GL_ARRAY_BUFFER, m_uiVBO);
	//PositionVertex* pVertices = (PositionVertex*)glMapBuffer(GL_ARRAY_BUFFER, GL_READ_WRITE);
	//assert(pVertices);	// TODO real error output?

	for(int i=0; i<iNumCols; ++i)
		for(int j=0; j<m_iNumRows; ++j)
		{
			// Integrate now this vertex one step
			m_pPositionMap[i*m_iNumRows+j].vPosition = _pMesh->Integrate(m_pPositionMap[i*m_iNumRows+j].vPosition, _fStepSize, AmiraMesh::INTEGRATION_MODEULER | AmiraMesh::INTEGRATION_FILTER_POINT);
			//pVertices[i*m_iNumRows+j].vPosition = _pMesh->Integrate(pVertices[i*m_iNumRows+j].vPosition, _fStepSize, AmiraMesh::INTEGRATION_MODEULER);
		}

	// Upoad as vertexmap
	//glUnmapBuffer(GL_ARRAY_BUFFER);
		glGetError();

	glBindTexture(GL_TEXTURE_2D, m_uiVertexMap);
	glTexImage2D(GL_TEXTURE_2D,	// Target
		0,						// Mip-Level
		GL_RGB32F,				// Internal format
		m_iNumRows,				// Width
		m_iNumCols,				// Height
		0,						// Border
		GL_RGB,					// Format
		GL_FLOAT,				// Type
		m_pPositionMap);		// Data

	const GLenum ErrorValue = glGetError();
	if(ErrorValue != GL_NO_ERROR) 
		printf("Vertexmap: %s\n",gluErrorString(ErrorValue));
}

int SmokeSurface::GetVBO()
{
	return m_uiVBO;
}

int SmokeSurface::GetNumColums()
{
	return m_iNumCols;
}

int SmokeSurface::GetNumRows()
{
	return m_iNumRows;
}

int SmokeSurface::GetNumVertices()
{
	return m_iNumVertices;
}