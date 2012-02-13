#include <cassert>
#include <GL/glew.h>
#include <cstdlib>
#include "globals.hpp"
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
	m_bInvalidSeedLine = false;

	// Create data
	m_iNumVertices = _iNumCols*_iNumRows;
	int iVertexDataSize = sizeof(GridVertex)*m_iNumVertices;
	m_pPositionMap = (PositionVertex*)malloc(sizeof(PositionVertex)*m_iNumVertices);
	GridVertex* pGridVertices = (GridVertex*)malloc(iVertexDataSize);
	for(int i=0; i<_iNumCols; ++i)
		for(int j=0; j<_iNumRows; ++j)
		{
			pGridVertices[i*_iNumRows+j].fRow = j/(float)(_iNumRows-1);
			pGridVertices[i*_iNumRows+j].fColumn = i/(float)(_iNumCols-1);
			m_pPositionMap[i*_iNumRows+j].vPosition = glm::mix(_vStart, _vEnd, j/(float)(_iNumRows-1));
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

	// Insert triangulation
	glGenBuffers(1, &m_uiIBO);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_uiIBO);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, m_iNumIndices*sizeof(GLuint), pIndices, GL_STATIC_DRAW);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);

	// Create OpenGL buffers
	// Create Vertex array object
	glGenVertexArrays(1, &m_uiVAO);
	glBindVertexArray(m_uiVAO);
		// Create Vertex buffer object
		glGenBuffers(1, &m_uiVBO);
		glBindBuffer(GL_ARRAY_BUFFER, m_uiVBO);
		// Insert data and usage declaration
		glBufferData(GL_ARRAY_BUFFER, iVertexDataSize, pGridVertices, GL_STATIC_DRAW);
		glVertexAttribPointer(GLGraphics::ASLOT_POSITION, 2, GL_FLOAT, GL_FALSE, 0,0);
		glBindBuffer(GL_ARRAY_BUFFER,0);

		glEnableVertexAttribArray(GLGraphics::ASLOT_POSITION);

	glBindVertexArray(0);
	
	// data is loaded to GPU now
	free(pGridVertices);
	free(pIndices);

	//Create a pbo for a performant access through CUDA and copying the modified vertices to a texture completely on the GPU.
	glGenBuffers(1,&m_uiPBO);
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER,m_uiPBO);
	glBufferData(GL_PIXEL_UNPACK_BUFFER,m_iNumRows*m_iNumCols*sizeof(float)*3,reinterpret_cast<float*>(m_pPositionMap),GL_STATIC_COPY);

		glGenTextures(1, &m_uiVertexMap);
		glBindTexture(GL_TEXTURE_2D,m_uiVertexMap);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_S,GL_MIRRORED_REPEAT);
		glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_WRAP_T,GL_MIRRORED_REPEAT);

		glTexImage2D(GL_TEXTURE_2D,	// Target
			0,						// Mip-Level
			GL_RGB32F,				// Internal format
			m_iNumRows,				// Width
			m_iNumCols,				// Height
			0,						// Border
			GL_RGB,					// Format
			GL_FLOAT,				// Type
			NULL);					// Data
		glBindTexture(GL_TEXTURE_2D,0);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);

	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D,m_uiVertexMap);
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

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER,0);
	glDeleteBuffers(1, &m_uiPBO);

	// Detach and delete array
	glBindVertexArray(0);
	glDeleteVertexArrays(1, &m_uiVAO);

	glBindTexture(GL_TEXTURE_2D,m_uiVertexMap);
	glDeleteTextures(1,&m_uiVertexMap);

	free(m_pPositionMap);
}

// **************************************************************** //
// Set the buffers and make the rendercall for the geometry
void SmokeSurface::Render()
{
	glBindVertexArray(m_uiVAO);
	//glDrawArrays(GL_POINTS, 0, m_iNumVertices);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER,m_uiIBO);
	glDrawElements(GL_TRIANGLES, m_iNumIndices, GL_UNSIGNED_INT, (GLvoid*)0);
}

// **************************************************************** //
// Sets the next column to the seed line and increase the number of columns
void SmokeSurface::ReleaseNextColumn()
{
	// Set a base line (if in flow do nothing otherwise).
	if(m_iNumReleasedColumns >= m_iNumCols)
	{
		int i=m_iNumReleasedColumns%m_iNumCols;
		for(int j=0; j<m_iNumRows; ++j)
			m_pPositionMap[i*m_iNumRows+j].vPosition = glm::mix(m_vStart, m_vEnd, j/(float)(m_iNumRows-1));
	}
	++m_iNumReleasedColumns;
}

// **************************************************************** //
// Input:	_pMesh - vector field that should be visualized
//			_fStepSize - The size (distance) of the one integration step.
//				To integrate over larger timeslices call IntegrateCPU
//				multiple times.
void SmokeSurface::IntegrateCPU(AmiraMesh* _pMesh, float _fStepSize, int _iMethod)
{
	// Integrate everything in flow (streak surface)
	int iNumCols = glm::min(m_iNumCols, m_iNumReleasedColumns);

	for(int i=0; i<iNumCols; ++i)
		if(i!=m_iNumReleasedColumns%m_iNumCols)
		for(int j=0; j<m_iNumRows; ++j)
		{
			// Integrate now this vertex one step
			m_pPositionMap[i*m_iNumRows+j].vPosition = _pMesh->Integrate(
				m_pPositionMap[i*m_iNumRows+j].vPosition,
				_fStepSize,
				_iMethod);
		}

	// Upoad as vertexmap
	glGetError();	// Clear errors

	const GLenum ErrorValue = glGetError();
	if(ErrorValue != GL_NO_ERROR) 
		printf("Vertexmap: %s\n",gluErrorString(ErrorValue));
}

int SmokeSurface::GetVBO()
{
	return m_uiVBO;
}

int SmokeSurface::GetNumColumns()
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

// **************************************************************** //
// Reset all vertices to the seedline
void SmokeSurface::Reset()
{
	m_iNumReleasedColumns = 0;
	for(int i=0; i<m_iNumCols; ++i)
		for(int j=0; j<m_iNumRows; ++j)
			m_pPositionMap[i*m_iNumRows+j].vPosition = glm::mix(m_vStart, m_vEnd, j/(float)(m_iNumRows-1));
}