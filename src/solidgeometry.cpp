#include <cassert>
#include <GL/glew.h>
#include "solidgeometry.hpp"

// **************************************************************** //
// Marching cubes indices (+ vertices)
//										
//	y				4+---------------4---------------+5
//	|				/|                              /|
//	|			  7  |                            5  |
//	|			 /   8                           /   9
//	|(0,0,0)  7+---------------6---------------+6    |
//	|		   |     |                         |     |
//	|		   |    0+--------------0----------|-----+1	(1,1,1)
//	|	   z   11   /                          10   /
//	|	 /	   |  3                            |  1
//	|  /	   |/                              |/
//	|/		  3+---------------2---------------+2
//	+-----------------------------------------------------> x
//
// Neighborhoods:
// To optimize the number of vertices reuse these one which were
// created in an previos cell. Therefore we have to find the
// correct neighbor. For each edge are four cells adjacent, but
// only one of them was the first seen. The vertex was only added
// one time.
// Scince we traverse the data x innermost and z outermost the neighbor
// in z direction was the first one. So the cells have the priorities z-y-x.
// To calculate the edge index for this neighbor cell use the
// g_acNeighborEdge lookup table.

// **************************************************************** //
// Lookup tables for marching cubes
// Tables by Cory Gene Bloyd (from http://paulbourke.net/geometry/polygonise/)
// Edges contains flags which edges are cut/have a vertex.
unsigned short g_acEdges[256] = {
		0x0  , 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
		0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
		0x190, 0x99 , 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
		0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
		0x230, 0x339, 0x33 , 0x13a, 0x636, 0x73f, 0x435, 0x53c,
		0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
		0x3a0, 0x2a9, 0x1a3, 0xaa , 0x7a6, 0x6af, 0x5a5, 0x4ac,
		0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
		0x460, 0x569, 0x663, 0x76a, 0x66 , 0x16f, 0x265, 0x36c,
		0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
		0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff , 0x3f5, 0x2fc,
		0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
		0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55 , 0x15c,
		0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
		0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc ,
		0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
		0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
		0xcc , 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
		0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
		0x15c, 0x55 , 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
		0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
		0x2fc, 0x3f5, 0xff , 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
		0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
		0x36c, 0x265, 0x16f, 0x66 , 0x76a, 0x663, 0x569, 0x460,
		0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
		0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa , 0x1a3, 0x2a9, 0x3a0,
		0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
		0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33 , 0x339, 0x230,
		0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
		0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99 , 0x190,
		0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
		0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0   };

// **************************************************************** //
// Position offset vectors for all 12 edges (cell midpoint -> edge midpoint)
const glm::vec3 g_vEdgeOffsets[] = {glm::vec3(0.0f, 0.5f, 0.5f),
									glm::vec3(0.5f, 0.5f, 0.0f),
									glm::vec3(0.0f, 0.5f, -0.5f),
									glm::vec3(-0.5f, 0.5f, 0.0f),
									glm::vec3(0.0f, -0.5f, 0.5f),
									glm::vec3(0.5f, -0.5f, 0.0f),
									glm::vec3(0.0f, -0.5f, -0.5f),
									glm::vec3(-0.5f, -0.5f, 0.0f),
									glm::vec3(-0.5f, 0.0f, 0.5f),
									glm::vec3(0.5f, 0.0f, 0.5f),
									glm::vec3(0.5f, 0.0f, -0.5f),
									glm::vec3(-0.5f, 0.0f, -0.5f)	};
// Exclude edge-vertices from creation if in a middle position
const unsigned int g_uiVertexCreationMaskX = 0x677;	// 011001110111
const unsigned int g_uiVertexCreationMaskY = 0xf0f;	// 111100001111
const unsigned int g_uiVertexCreationMaskZ = 0x3bb;	// 001110111011

// **************************************************************** //
// TriangleIndices contains the index offsets for the triangles.
// The offset 0 refers to the vertex of the first (least significant) egde.
// Therefore the indices depend on the creation order of vertices.
/*char g_acTriangleIndices[256][15] =
	   {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  2,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  1,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  2,  1,  3,  2,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  1,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  4,  3,  1,  2,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 2,  1,  3,  0,  1,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  2,  1,  0,  4,  2,  4,  3,  2, -1, -1, -1, -1, -1, -1},
		{ 1,  2,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  3,  1,  2,  3,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 1,  4,  0,  2,  3,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  4,  1,  0,  3,  4,  3,  2,  4, -1, -1, -1, -1, -1, -1},
		{ 1,  2,  0,  3,  2,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  3,  1,  0,  2,  3,  2,  4,  3, -1, -1, -1, -1, -1, -1},
		{ 1,  2,  0,  1,  4,  2,  4,  3,  2, -1, -1, -1, -1, -1, -1},
		{ 1,  0,  2,  2,  0,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  1,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 2,  1,  0,  3,  1,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  1 , 5,  4,  2,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 2,  0,  4,  2,  3,  0,  3,  1,  0, -1, -1, -1, -1, -1, -1},
		{ 0,  1,  5,  4,  2,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 3,  4,  5,  3,  0,  4,  1,  2,  6, -1, -1, -1, -1, -1, -1},
		{ 5,  1,  6,  5,  0,  1,  4,  2,  3, -1, -1, -1, -1, -1, -1},
		{ 0,  5,  4,  0,  4,  3,  0,  3,  1,  3,  4,  2, -1, -1, -1},
		{ 4,  2,  3,  1,  5,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 4,  2,  3,  4,  1,  2,  1,  0,  2, -1, -1, -1, -1, -1, -1},
		{ 7,  0,  1,  6,  4,  5,  2,  3,  8, -1, -1, -1, -1, -1, -1},
		{ 2,  3,  5,  4,  2,  5,  4,  5,  1,  4,  1,  0, -1, -1, -1},
		{ 1,  5,  0,  1,  6,  5,  3,  4,  2, -1, -1, -1, -1, -1, -1},
		{ 1,  5,  4,  1,  2,  5,  1,  0,  2,  3,  5,  2, -1, -1, -1},
		{ 2,  3,  4,  5,  0,  7,  5,  7,  6,  7,  0,  1, -1, -1, -1},
		{ 0,  1,  4,  0,  4,  2,  2,  4,  3, -1, -1, -1, -1, -1, -1},
		{ 2,  1,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 5,  3,  2,  0,  4,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  3,  2,  1,  3,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 4,  3,  2,  4,  1,  3,  1,  0,  3, -1, -1, -1, -1, -1, -1},
		{ 0,  1,  5,  4,  3,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 3,  0,  6,  1,  2,  8,  4,  7,  5, -1, -1, -1, -1, -1, -1},
		{ 3,  1,  4,  3,  2,  1,  2,  0,  1, -1, -1, -1, -1, -1, -1},
		{ 0,  5,  3,  1,  0,  3,  1,  3,  2,  1,  2,  4, -1, -1, -1},
		{ 4,  3,  2,  0,  1,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  6,  1,  0,  4,  6,  2,  5,  3, -1, -1, -1, -1, -1, -1},
		{ 0,  5,  4,  0,  1,  5,  2,  3,  6, -1, -1, -1, -1, -1, -1},
		{ 1,  0,  3,  1,  3,  4,  1,  4,  5,  2,  4,  3, -1, -1, -1},*/

char g_acTriangleIndices[256][15] =
	   {{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  1,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 1,  8,  3,  9,  8,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  8,  3,  1,  2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 9,  2, 10,  0,  2,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 2,  8,  3,  2, 10,  8, 10,  9,  8, -1, -1, -1, -1, -1, -1},
		{ 3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0, 11,  2,  8, 11,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 1,  9,  0,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 1, 11,  2,  1,  9, 11,  9,  8, 11, -1, -1, -1, -1, -1, -1},
		{ 3, 10,  1, 11, 10,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0, 10,  1,  0,  8, 10,  8, 11, 10, -1, -1, -1, -1, -1, -1},
		{ 3,  9,  0,  3, 11,  9, 11, 10,  9, -1, -1, -1, -1, -1, -1},
		{ 9,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 4,  3,  0,  7,  3,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  1,  9,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 4,  1,  9,  4,  7,  1,  7,  3,  1, -1, -1, -1, -1, -1, -1},
		{ 1,  2, 10,  8,  4,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 3,  4,  7,  3,  0,  4,  1,  2, 10, -1, -1, -1, -1, -1, -1},
		{ 9,  2, 10,  9,  0,  2,  8,  4,  7, -1, -1, -1, -1, -1, -1},
		{ 2, 10,  9,  2,  9,  7,  2,  7,  3,  7,  9,  4, -1, -1, -1},
		{ 8,  4,  7,  3, 11,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11,  4,  7, 11,  2,  4,  2,  0,  4, -1, -1, -1, -1, -1, -1},
		{ 9,  0,  1,  8,  4,  7,  2,  3, 11, -1, -1, -1, -1, -1, -1},
		{ 4,  7, 11,  9,  4, 11,  9, 11,  2,  9,  2,  1, -1, -1, -1},
		{ 3, 10,  1,  3, 11, 10,  7,  8,  4, -1, -1, -1, -1, -1, -1},
		{ 1, 11, 10,  1,  4, 11,  1,  0,  4,  7, 11,  4, -1, -1, -1},
		{ 4,  7,  8,  9,  0, 11,  9, 11, 10, 11,  0,  3, -1, -1, -1},
		{ 4,  7, 11,  4, 11,  9,  9, 11, 10, -1, -1, -1, -1, -1, -1},
		{ 9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 9,  5,  4,  0,  8,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  5,  4,  1,  5,  0, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 8,  5,  4,  8,  3,  5,  3,  1,  5, -1, -1, -1, -1, -1, -1},
		{ 1,  2, 10,  9,  5,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 3,  0,  8,  1,  2, 10,  4,  9,  5, -1, -1, -1, -1, -1, -1},
		{ 5,  2, 10,  5,  4,  2,  4,  0,  2, -1, -1, -1, -1, -1, -1},
		{ 2, 10,  5,  3,  2,  5,  3,  5,  4,  3,  4,  8, -1, -1, -1},
		{ 9,  5,  4,  2,  3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0, 11,  2,  0,  8, 11,  4,  9,  5, -1, -1, -1, -1, -1, -1},
		{ 0,  5,  4,  0,  1,  5,  2,  3, 11, -1, -1, -1, -1, -1, -1},
		{ 2,  1,  5,  2,  5,  8,  2,  8, 11,  4,  8,  5, -1, -1, -1},

		{10,  3, 11, 10,  1,  3,  9,  5,  4, -1, -1, -1, -1, -1, -1},
		{ 4,  9,  5,  0,  8,  1,  8, 10,  1,  8, 11, 10, -1, -1, -1},
		{ 5,  4,  0,  5,  0, 11,  5, 11, 10, 11,  0,  3, -1, -1, -1},
		{ 5,  4,  8,  5,  8, 10, 10,  8, 11, -1, -1, -1, -1, -1, -1},
		{ 9,  7,  8,  5,  7,  9, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 9,  3,  0,  9,  5,  3,  5,  7,  3, -1, -1, -1, -1, -1, -1},
		{ 0,  7,  8,  0,  1,  7,  1,  5,  7, -1, -1, -1, -1, -1, -1},
		{ 1,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 9,  7,  8,  9,  5,  7, 10,  1,  2, -1, -1, -1, -1, -1, -1},
		{10,  1,  2,  9,  5,  0,  5,  3,  0,  5,  7,  3, -1, -1, -1},
		{ 8,  0,  2,  8,  2,  5,  8,  5,  7, 10,  5,  2, -1, -1, -1},
		{ 2, 10,  5,  2,  5,  3,  3,  5,  7, -1, -1, -1, -1, -1, -1},
		{ 7,  9,  5,  7,  8,  9,  3, 11,  2, -1, -1, -1, -1, -1, -1},
		{ 9,  5,  7,  9,  7,  2,  9,  2,  0,  2,  7, 11, -1, -1, -1},
		{ 2,  3, 11,  0,  1,  8,  1,  7,  8,  1,  5,  7, -1, -1, -1},
		{11,  2,  1, 11,  1,  7,  7,  1,  5, -1, -1, -1, -1, -1, -1},
		{ 9,  5,  8,  8,  5,  7, 10,  1,  3, 10,  3, 11, -1, -1, -1},
		{ 5,  7,  0,  5,  0,  9,  7, 11,  0,  1,  0, 10, 11, 10,  0},
		{11, 10,  0, 11,  0,  3, 10,  5,  0,  8,  0,  7,  5,  7,  0},
		{11, 10,  5,  7, 11,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  8,  3,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 9,  0,  1,  5, 10,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 1,  8,  3,  1,  9,  8,  5, 10,  6, -1, -1, -1, -1, -1, -1},
		{ 1,  6,  5,  2,  6,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 1,  6,  5,  1,  2,  6,  3,  0,  8, -1, -1, -1, -1, -1, -1},
		{ 9,  6,  5,  9,  0,  6,  0,  2,  6, -1, -1, -1, -1, -1, -1},
		{ 5,  9,  8,  5,  8,  2,  5,  2,  6,  3,  2,  8, -1, -1, -1},
		{ 2,  3, 11, 10,  6,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11,  0,  8, 11,  2,  0, 10,  6,  5, -1, -1, -1, -1, -1, -1},
		{ 0,  1,  9,  2,  3, 11,  5, 10,  6, -1, -1, -1, -1, -1, -1},
		{ 5, 10,  6,  1,  9,  2,  9, 11,  2,  9,  8, 11, -1, -1, -1},
		{ 6,  3, 11,  6,  5,  3,  5,  1,  3, -1, -1, -1, -1, -1, -1},
		{ 0,  8, 11,  0, 11,  5,  0,  5,  1,  5, 11,  6, -1, -1, -1},
		{ 3, 11,  6,  0,  3,  6,  0,  6,  5,  0,  5,  9, -1, -1, -1},
		{ 6,  5,  9,  6,  9, 11, 11,  9,  8, -1, -1, -1, -1, -1, -1},
		{ 5, 10,  6,  4,  7,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 4,  3,  0,  4,  7,  3,  6,  5, 10, -1, -1, -1, -1, -1, -1},
		{ 1,  9,  0,  5, 10,  6,  8,  4,  7, -1, -1, -1, -1, -1, -1},
		{10,  6,  5,  1,  9,  7,  1,  7,  3,  7,  9,  4, -1, -1, -1},
		{ 6,  1,  2,  6,  5,  1,  4,  7,  8, -1, -1, -1, -1, -1, -1},
		{ 1,  2,  5,  5,  2,  6,  3,  0,  4,  3,  4,  7, -1, -1, -1},
		{ 8,  4,  7,  9,  0,  5,  0,  6,  5,  0,  2,  6, -1, -1, -1},
		{ 7,  3,  9,  7,  9,  4,  3,  2,  9,  5,  9,  6,  2,  6,  9},
		{ 3, 11,  2,  7,  8,  4, 10,  6,  5, -1, -1, -1, -1, -1, -1},
		{ 5, 10,  6,  4,  7,  2,  4,  2,  0,  2,  7, 11, -1, -1, -1},
		{ 0,  1,  9,  4,  7,  8,  2,  3, 11,  5, 10,  6, -1, -1, -1},
		{ 9,  2,  1,  9, 11,  2,  9,  4, 11,  7, 11,  4,  5, 10,  6},
		{ 8,  4,  7,  3, 11,  5,  3,  5,  1,  5, 11,  6, -1, -1, -1},
		{ 5,  1, 11,  5, 11,  6,  1,  0, 11,  7, 11,  4,  0,  4, 11},
		{ 0,  5,  9,  0,  6,  5,  0,  3,  6, 11,  6,  3,  8,  4,  7},
		{ 6,  5,  9,  6,  9, 11,  4,  7,  9,  7, 11,  9, -1, -1, -1},
		{10,  4,  9,  6,  4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 4, 10,  6,  4,  9, 10,  0,  8,  3, -1, -1, -1, -1, -1, -1},
		{10,  0,  1, 10,  6,  0,  6,  4,  0, -1, -1, -1, -1, -1, -1},
		{ 8,  3,  1,  8,  1,  6,  8,  6,  4,  6,  1, 10, -1, -1, -1},
		{ 1,  4,  9,  1,  2,  4,  2,  6,  4, -1, -1, -1, -1, -1, -1},
		{ 3,  0,  8,  1,  2,  9,  2,  4,  9,  2,  6,  4, -1, -1, -1},
		{ 0,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 8,  3,  2,  8,  2,  4,  4,  2,  6, -1, -1, -1, -1, -1, -1},
		{10,  4,  9, 10,  6,  4, 11,  2,  3, -1, -1, -1, -1, -1, -1},
		{ 0,  8,  2,  2,  8, 11,  4,  9, 10,  4, 10,  6, -1, -1, -1},
		{ 3, 11,  2,  0,  1,  6,  0,  6,  4,  6,  1, 10, -1, -1, -1},
		{ 6,  4,  1,  6,  1, 10,  4,  8,  1,  2,  1, 11,  8, 11,  1},
		{ 9,  6,  4,  9,  3,  6,  9,  1,  3, 11,  6,  3, -1, -1, -1},
		{ 8, 11,  1,  8,  1,  0, 11,  6,  1,  9,  1,  4,  6,  4,  1},
		{ 3, 11,  6,  3,  6,  0,  0,  6,  4, -1, -1, -1, -1, -1, -1},
		{ 6,  4,  8, 11,  6,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 7, 10,  6,  7,  8, 10,  8,  9, 10, -1, -1, -1, -1, -1, -1},
		{ 0,  7,  3,  0, 10,  7,  0,  9, 10,  6,  7, 10, -1, -1, -1},
		{10,  6,  7,  1, 10,  7,  1,  7,  8,  1,  8,  0, -1, -1, -1},
		{10,  6,  7, 10,  7,  1,  1,  7,  3, -1, -1, -1, -1, -1, -1},
		{ 1,  2,  6,  1,  6,  8,  1,  8,  9,  8,  6,  7, -1, -1, -1},
		{ 2,  6,  9,  2,  9,  1,  6,  7,  9,  0,  9,  3,  7,  3,  9},
		{ 7,  8,  0,  7,  0,  6,  6,  0,  2, -1, -1, -1, -1, -1, -1},
		{ 7,  3,  2,  6,  7,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 2,  3, 11, 10,  6,  8, 10,  8,  9,  8,  6,  7, -1, -1, -1},
		{ 2,  0,  7,  2,  7, 11,  0,  9,  7,  6,  7, 10,  9, 10,  7},
		{ 1,  8,  0,  1,  7,  8,  1, 10,  7,  6,  7, 10,  2,  3, 11},
		{11,  2,  1, 11,  1,  7, 10,  6,  1,  6,  7,  1, -1, -1, -1},
		{ 8,  9,  6,  8,  6,  7,  9,  1,  6, 11,  6,  3,  1,  3,  6},
		{ 0,  9,  1, 11,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 7,  8,  0,  7,  0,  6,  3, 11,  0, 11,  6,  0, -1, -1, -1},
		{ 7, 11,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 3,  0,  8, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  1,  9, 11,  7,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 8,  1,  9,  8,  3,  1, 11,  7,  6, -1, -1, -1, -1, -1, -1},
		{10,  1,  2,  6, 11,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 1,  2, 10,  3,  0,  8,  6, 11,  7, -1, -1, -1, -1, -1, -1},
		{ 2,  9,  0,  2, 10,  9,  6, 11,  7, -1, -1, -1, -1, -1, -1},
		{ 6, 11,  7,  2, 10,  3, 10,  8,  3, 10,  9,  8, -1, -1, -1},
		{ 7,  2,  3,  6,  2,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 7,  0,  8,  7,  6,  0,  6,  2,  0, -1, -1, -1, -1, -1, -1},
		{ 2,  7,  6,  2,  3,  7,  0,  1,  9, -1, -1, -1, -1, -1, -1},
		{ 1,  6,  2,  1,  8,  6,  1,  9,  8,  8,  7,  6, -1, -1, -1},
		{10,  7,  6, 10,  1,  7,  1,  3,  7, -1, -1, -1, -1, -1, -1},
		{10,  7,  6,  1,  7, 10,  1,  8,  7,  1,  0,  8, -1, -1, -1},
		{ 0,  3,  7,  0,  7, 10,  0, 10,  9,  6, 10,  7, -1, -1, -1},
		{ 7,  6, 10,  7, 10,  8,  8, 10,  9, -1, -1, -1, -1, -1, -1},
		{ 6,  8,  4, 11,  8,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 3,  6, 11,  3,  0,  6,  0,  4,  6, -1, -1, -1, -1, -1, -1},
		{ 8,  6, 11,  8,  4,  6,  9,  0,  1, -1, -1, -1, -1, -1, -1},
		{ 9,  4,  6,  9,  6,  3,  9,  3,  1, 11,  3,  6, -1, -1, -1},
		{ 6,  8,  4,  6, 11,  8,  2, 10,  1, -1, -1, -1, -1, -1, -1},
		{ 1,  2, 10,  3,  0, 11,  0,  6, 11,  0,  4,  6, -1, -1, -1},
		{ 4, 11,  8,  4,  6, 11,  0,  2,  9,  2, 10,  9, -1, -1, -1},
		{10,  9,  3, 10,  3,  2,  9,  4,  3, 11,  3,  6,  4,  6,  3},
		{ 8,  2,  3,  8,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1},
		{ 0,  4,  2,  4,  6,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 1,  9,  0,  2,  3,  4,  2,  4,  6,  4,  3,  8, -1, -1, -1},
		{ 1,  9,  4,  1,  4,  2,  2,  4,  6, -1, -1, -1, -1, -1, -1},
		{ 8,  1,  3,  8,  6,  1,  8,  4,  6,  6, 10,  1, -1, -1, -1},
		{10,  1,  0, 10,  0,  6,  6,  0,  4, -1, -1, -1, -1, -1, -1},
		{ 4,  6,  3,  4,  3,  8,  6, 10,  3,  0,  3,  9, 10,  9,  3},
		{10,  9,  4,  6, 10,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 4,  9,  5,  7,  6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  8,  3,  4,  9,  5, 11,  7,  6, -1, -1, -1, -1, -1, -1},
		{ 5,  0,  1,  5,  4,  0,  7,  6, 11, -1, -1, -1, -1, -1, -1},
		{11,  7,  6,  8,  3,  4,  3,  5,  4,  3,  1,  5, -1, -1, -1},
		{ 9,  5,  4, 10,  1,  2,  7,  6, 11, -1, -1, -1, -1, -1, -1},
		{ 6, 11,  7,  1,  2, 10,  0,  8,  3,  4,  9,  5, -1, -1, -1},
		{ 7,  6, 11,  5,  4, 10,  4,  2, 10,  4,  0,  2, -1, -1, -1},
		{ 3,  4,  8,  3,  5,  4,  3,  2,  5, 10,  5,  2, 11,  7,  6},
		{ 7,  2,  3,  7,  6,  2,  5,  4,  9, -1, -1, -1, -1, -1, -1},
		{ 9,  5,  4,  0,  8,  6,  0,  6,  2,  6,  8,  7, -1, -1, -1},
		{ 3,  6,  2,  3,  7,  6,  1,  5,  0,  5,  4,  0, -1, -1, -1},
		{ 6,  2,  8,  6,  8,  7,  2,  1,  8,  4,  8,  5,  1,  5,  8},
		{ 9,  5,  4, 10,  1,  6,  1,  7,  6,  1,  3,  7, -1, -1, -1},
		{ 1,  6, 10,  1,  7,  6,  1,  0,  7,  8,  7,  0,  9,  5,  4},
		{ 4,  0, 10,  4, 10,  5,  0,  3, 10,  6, 10,  7,  3,  7, 10},
		{ 7,  6, 10,  7, 10,  8,  5,  4, 10,  4,  8, 10, -1, -1, -1},
		{ 6,  9,  5,  6, 11,  9, 11,  8,  9, -1, -1, -1, -1, -1, -1},
		{ 3,  6, 11,  0,  6,  3,  0,  5,  6,  0,  9,  5, -1, -1, -1},
		{ 0, 11,  8,  0,  5, 11,  0,  1,  5,  5,  6, 11, -1, -1, -1},
		{ 6, 11,  3,  6,  3,  5,  5,  3,  1, -1, -1, -1, -1, -1, -1},
		{ 1,  2, 10,  9,  5, 11,  9, 11,  8, 11,  5,  6, -1, -1, -1},
		{ 0, 11,  3,  0,  6, 11,  0,  9,  6,  5,  6,  9,  1,  2, 10},
		{11,  8,  5, 11,  5,  6,  8,  0,  5, 10,  5,  2,  0,  2,  5},
		{ 6, 11,  3,  6,  3,  5,  2, 10,  3, 10,  5,  3, -1, -1, -1},
		{ 5,  8,  9,  5,  2,  8,  5,  6,  2,  3,  8,  2, -1, -1, -1},
		{ 9,  5,  6,  9,  6,  0,  0,  6,  2, -1, -1, -1, -1, -1, -1},
		{ 1,  5,  8,  1,  8,  0,  5,  6,  8,  3,  8,  2,  6,  2,  8},
		{ 1,  5,  6,  2,  1,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 1,  3,  6,  1,  6, 10,  3,  8,  6,  5,  6,  9,  8,  9,  6},
		{10,  1,  0, 10,  0,  6,  9,  5,  0,  5,  6,  0, -1, -1, -1},
		{ 0,  3,  8,  5,  6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{10,  5,  6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11,  5, 10,  7,  5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{11,  5, 10, 11,  7,  5,  8,  3,  0, -1, -1, -1, -1, -1, -1},
		{ 5, 11,  7,  5, 10, 11,  1,  9,  0, -1, -1, -1, -1, -1, -1},
		{10,  7,  5, 10, 11,  7,  9,  8,  1,  8,  3,  1, -1, -1, -1},
		{11,  1,  2, 11,  7,  1,  7,  5,  1, -1, -1, -1, -1, -1, -1},
		{ 0,  8,  3,  1,  2,  7,  1,  7,  5,  7,  2, 11, -1, -1, -1},
		{ 9,  7,  5,  9,  2,  7,  9,  0,  2,  2, 11,  7, -1, -1, -1},
		{ 7,  5,  2,  7,  2, 11,  5,  9,  2,  3,  2,  8,  9,  8,  2},
		{ 2,  5, 10,  2,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1},
		{ 8,  2,  0,  8,  5,  2,  8,  7,  5, 10,  2,  5, -1, -1, -1},
		{ 9,  0,  1,  5, 10,  3,  5,  3,  7,  3, 10,  2, -1, -1, -1},
		{ 9,  8,  2,  9,  2,  1,  8,  7,  2, 10,  2,  5,  7,  5,  2},
		{ 1,  3,  5,  3,  7,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  8,  7,  0,  7,  1,  1,  7,  5, -1, -1, -1, -1, -1, -1},
		{ 9,  0,  3,  9,  3,  5,  5,  3,  7, -1, -1, -1, -1, -1, -1},
		{ 9,  8,  7,  5,  9,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 5,  8,  4,  5, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1},
		{ 5,  0,  4,  5, 11,  0,  5, 10, 11, 11,  3,  0, -1, -1, -1},
		{ 0,  1,  9,  8,  4, 10,  8, 10, 11, 10,  4,  5, -1, -1, -1},
		{10, 11,  4, 10,  4,  5, 11,  3,  4,  9,  4,  1,  3,  1,  4},
		{ 2,  5,  1,  2,  8,  5,  2, 11,  8,  4,  5,  8, -1, -1, -1},
		{ 0,  4, 11,  0, 11,  3,  4,  5, 11,  2, 11,  1,  5,  1, 11},
		{ 0,  2,  5,  0,  5,  9,  2, 11,  5,  4,  5,  8, 11,  8,  5},
		{ 9,  4,  5,  2, 11,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 2,  5, 10,  3,  5,  2,  3,  4,  5,  3,  8,  4, -1, -1, -1},
		{ 5, 10,  2,  5,  2,  4,  4,  2,  0, -1, -1, -1, -1, -1, -1},
		{ 3, 10,  2,  3,  5, 10,  3,  8,  5,  4,  5,  8,  0,  1,  9},
		{ 5, 10,  2,  5,  2,  4,  1,  9,  2,  9,  4,  2, -1, -1, -1},
		{ 8,  4,  5,  8,  5,  3,  3,  5,  1, -1, -1, -1, -1, -1, -1},
		{ 0,  4,  5,  1,  0,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 8,  4,  5,  8,  5,  3,  9,  0,  5,  0,  3,  5, -1, -1, -1},
		{ 9,  4,  5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 4, 11,  7,  4,  9, 11,  9, 10, 11, -1, -1, -1, -1, -1, -1},
		{ 0,  8,  3,  4,  9,  7,  9, 11,  7,  9, 10, 11, -1, -1, -1},
		{ 1, 10, 11,  1, 11,  4,  1,  4,  0,  7,  4, 11, -1, -1, -1},
		{ 3,  1,  4,  3,  4,  8,  1, 10,  4,  7,  4, 11, 10, 11,  4},
		{ 4, 11,  7,  9, 11,  4,  9,  2, 11,  9,  1,  2, -1, -1, -1},
		{ 9,  7,  4,  9, 11,  7,  9,  1, 11,  2, 11,  1,  0,  8,  3},
		{11,  7,  4, 11,  4,  2,  2,  4,  0, -1, -1, -1, -1, -1, -1},
		{11,  7,  4, 11,  4,  2,  8,  3,  4,  3,  2,  4, -1, -1, -1},
		{ 2,  9, 10,  2,  7,  9,  2,  3,  7,  7,  4,  9, -1, -1, -1},
		{ 9, 10,  7,  9,  7,  4, 10,  2,  7,  8,  7,  0,  2,  0,  7},
		{ 3,  7, 10,  3, 10,  2,  7,  4, 10,  1, 10,  0,  4,  0, 10},
		{ 1, 10,  2,  8,  7,  4, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 4,  9,  1,  4,  1,  7,  7,  1,  3, -1, -1, -1, -1, -1, -1},
		{ 4,  9,  1,  4,  1,  7,  0,  8,  1,  8,  7,  1, -1, -1, -1},
		{ 4,  0,  3,  7,  4,  3, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 4,  8,  7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 9, 10,  8, 10, 11,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 3,  0,  9,  3,  9, 11, 11,  9, 10, -1, -1, -1, -1, -1, -1},
		{ 0,  1, 10,  0, 10,  8,  8, 10, 11, -1, -1, -1, -1, -1, -1},
		{ 3,  1, 10, 11,  3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 1,  2, 11,  1, 11,  9,  9, 11,  8, -1, -1, -1, -1, -1, -1},
		{ 3,  0,  9,  3,  9, 11,  1,  2,  9,  2, 11,  9, -1, -1, -1},
		{ 0,  2, 11,  8,  0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 3,  2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 2,  3,  8,  2,  8, 10, 10,  8,  9, -1, -1, -1, -1, -1, -1},
		{ 9, 10,  2,  0,  9,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 2,  3,  8,  2,  8, 10,  0,  1,  8,  1, 10,  8, -1, -1, -1},
		{ 1, 10,  2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 1,  3,  8,  9,  1,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  9,  1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{ 0,  3,  8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
		{-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}};

// **************************************************************** //
// Parallel bit counting to calculate indices from edge numbers
const unsigned int MASK_01010101  = 0x55555555;
const unsigned int MASK_00110011  = 0x33333333;
const unsigned int MASK_00001111  = 0x0f0f0f0f;
const unsigned int MASK_0x8_1x8   = 0x00ff00ff;
const unsigned int MASK_0x16_1x16 = 0x0000ffff;
unsigned int CountBits32(unsigned int n)
{
	n =    (n & MASK_01010101)  + ((n >> 1) & MASK_01010101);
	n =    (n & MASK_00110011)  + ((n >> 2) & MASK_00110011);
	n =    (n & MASK_00001111)  + ((n >> 4) & MASK_00001111);
	n =    (n & MASK_0x8_1x8)   + ((n >> 8) & MASK_0x8_1x8);
	return (n & MASK_0x16_1x16) + ((n >> 16) & MASK_0x16_1x16);
}

unsigned int CountBits16(unsigned short n)
{
	n =    (n & MASK_01010101) + ((n >> 1) & MASK_01010101);
	n =	   (n & MASK_00110011) + ((n >> 2) & MASK_00110011);
	n =    (n & MASK_00001111) + ((n >> 4) & MASK_00001111);
	return (n & MASK_0x8_1x8)  + ((n >> 8) & MASK_0x8_1x8);
}

// **************************************************************** //
// The neighbors could have created vertices - which index offset
// has the vertex at the spezified edge?
char g_acNeighborEdge[6][12] =
	   {{ 2, -1, -1, -1,  6, -1, -1, -1, 11, 10, -1, -1},		// z+
		{ 4,  5,  6,  7, -1, -1, -1, -1, -1, -1, -1, -1},		// y+
		{-1,  3, -1, -1, -1,  7, -1, -1, -1,  8, 11, -1},		// x+
		{-1, -1,  0, -1, -1, -1,  4, -1, -1, -1,  9,  8},		// z-
		{-1, -1, -1, -1,  0,  1,  2,  3, -1, -1, -1, -1},		// y-
		{-1, -1, -1,  1, -1, -1, -1,  5,  9, -1, -1, 10}};		// x-

// Go back as much as possible in each direction. If there was no neighbor (borders of volume)
// in the direction look for an other possible older neighbor. If this doesn't exists go
// to the node itself (0,0,0) - no neighbor edge.
// Interpretation: (1,0,0) - use vertex from edge 1 from the left neighbor (x-1 - neighbor)
char g_acNeighborPriority[12][3] = 
{{-1,-1,-1},
 {-1,-1,-1},
 {-1,-1, 0},
 { 1,-1,-1},
 {-1, 0,-1},
 {-1, 1,-1},
 {-1, 0, 4},
 { 1, 3,-1},
 { 9,-1,-1},
 {-1,-1,-1},
 {-1,-1, 9},
 { 9,-1, 8}
};

// **************************************************************** //
// Calculate how many vertices are created before the one for the current edge.
// Cell index offset + edge offset = vertex index
// Input: _usLookUp - the edge splitting value from g_asEdges.
//		_uiEdge - the edge number (see picture of the box above)
unsigned int GetEdgeIndexOffset(unsigned short _usLookUp, unsigned int _uiEdge)
{
	return CountBits16(_usLookUp & ((1 << _uiEdge)-1));
}

// **************************************************************** //
// Vertex formats
struct SolidVertex {
	glm::vec3 vPosition;
	glm::vec3 vNormal;
};

// **************************************************************** //
// Sample the vector field and estimate solid or not
bool IsSolid(AmiraMesh* _pMesh, int _x, int _y, int _z)
{
	int iDataIndex = ((_z*_pMesh->GetSizeY()+_y)*_pMesh->GetSizeX()+_x)*3;
	return (abs(_pMesh->GetData()[iDataIndex])
		+ abs(_pMesh->GetData()[iDataIndex+1])
		+ abs(_pMesh->GetData()[iDataIndex+2]))
			<= 0.000001f;
}

// **************************************************************** //
// Create one surface mesh for the solid parts in a vector field.
SolidSurface::SolidSurface(AmiraMesh* _pMesh, int _iTriangles)
{
	// Create Buffers for the estimated number of triangles
	SolidVertex*  pVertexData = (SolidVertex*)malloc(sizeof(SolidVertex) * _iTriangles * 3);	// Upload only necessary vertices at the end, but worst case is every triangle has its own vertices (not possible - closed surface - worst case==valence 3?)
	unsigned int* pIndexData = (unsigned int*)malloc(sizeof(unsigned int) * _iTriangles * 3);
	int iBufferSize = sizeof(unsigned int) * (_pMesh->GetSizeX() * _pMesh->GetSizeY() - 1);
	unsigned int* pIndexOffsetsZ1 = (unsigned int*)malloc(iBufferSize);
	unsigned int* pIndexOffsetsZ0 = (unsigned int*)malloc(iBufferSize);
	memset(pIndexOffsetsZ1, 0, iBufferSize);
	unsigned int uiIndex = 0;
	unsigned int uiVertex = 0;

	glm::vec3 vDimension = _pMesh->GetBoundingBoxMax() - _pMesh->GetBoundingBoxMin();
	glm::vec3 vCellSize = glm::vec3(1.0f/_pMesh->GetSizeX(), 1.0f/_pMesh->GetSizeY(), 1.0f/_pMesh->GetSizeZ());

	// Traverse the amira mesh and create triangles with marching cubes.
	// There is one cube less than data points in each direction.
	for(int z=0;z<_pMesh->GetSizeZ()-1;++z)
	{
		for(int y=0;y<_pMesh->GetSizeY()-1;++y)
		{
			for(int x=0;x<_pMesh->GetSizeX()-1;++x)
			{
				// Create case-code
				unsigned int uiCase;
				uiCase  = IsSolid(_pMesh, x  , y+1, z+1);
				uiCase |= IsSolid(_pMesh, x+1, y+1, z+1)<<1;
				uiCase |= IsSolid(_pMesh, x+1, y+1, z  )<<2;
				uiCase |= IsSolid(_pMesh, x  , y+1, z  )<<3;
				uiCase |= IsSolid(_pMesh, x  , y  , z+1)<<4;
				uiCase |= IsSolid(_pMesh, x+1, y  , z+1)<<5;
				uiCase |= IsSolid(_pMesh, x+1, y  , z  )<<6;
				uiCase |= IsSolid(_pMesh, x  , y  , z  )<<7;

				// Save current index offset and case in neighborhood table (for later boxes)
				pIndexOffsetsZ0[y*_pMesh->GetSizeX()+x] = (uiCase<<24) | uiVertex;
				unsigned int uiV = uiVertex;

				// Lookup on which edges a split-vertex is generated
				int iEdges = g_acEdges[uiCase];
				// Exlude all prior created vertices
				if(x>0) iEdges &= g_uiVertexCreationMaskX;
				if(y>0) iEdges &= g_uiVertexCreationMaskY;
				if(z>0) iEdges &= g_uiVertexCreationMaskZ;

		//		if(uiVertex+12 >= _iTriangles) goto finishcreation;
				// Add vertices
				for(int i=0;i<12;++i)
				{
					if(iEdges & (1<<i))
					{
						// Calculate position exact in the middle (cell)
						pVertexData[uiVertex].vPosition.x = (x+0.5f)/(_pMesh->GetSizeX()-2);
						pVertexData[uiVertex].vPosition.y = (y+0.5f)/(_pMesh->GetSizeY()-2);
						pVertexData[uiVertex].vPosition.z = (z+0.5f)/(_pMesh->GetSizeZ()-2);
						// Calculate offset to the edge
						pVertexData[uiVertex].vPosition += g_vEdgeOffsets[i]*vCellSize;
						pVertexData[uiVertex].vPosition *= vDimension;
						pVertexData[uiVertex].vPosition += _pMesh->GetBoundingBoxMin();
//						printf("%f, %f, %f\n", pVertexData[uiVertex].vPosition.x, pVertexData[uiVertex].vPosition.y, pVertexData[uiVertex].vPosition.z);
						// Do something with the normal (TODO)
						++uiVertex;
					}
				}

				// Add indices
				int i=-1;
				while((i<14) && (g_acTriangleIndices[uiCase][++i]>-1))
				{
					// Calculate in which cell the vertex was created and on which edge.
					char cCurrentEdgeIndex = g_acTriangleIndices[uiCase][i];
					char cGotoX = (g_acNeighborPriority[cCurrentEdgeIndex][0]>=0) && (x>0);
					char cGotoY = (g_acNeighborPriority[cCurrentEdgeIndex][1]>=0) && (y>0);
					char cGotoZ = (g_acNeighborPriority[cCurrentEdgeIndex][2]>=0) && (z>0);
					char cNeighborEdge = cCurrentEdgeIndex;
					if(cGotoZ) cNeighborEdge = g_acNeighborEdge[3][cCurrentEdgeIndex];
					if(cGotoY) cNeighborEdge = g_acNeighborEdge[4][cCurrentEdgeIndex];
					if(cGotoX) cNeighborEdge = g_acNeighborEdge[5][cCurrentEdgeIndex];
					if(cGotoZ && cGotoX) cNeighborEdge = g_acNeighborPriority[cCurrentEdgeIndex][0];
					if(cGotoZ && cGotoY) cNeighborEdge = g_acNeighborPriority[cCurrentEdgeIndex][1];
					if(cGotoY && cGotoX) cNeighborEdge = g_acNeighborPriority[cCurrentEdgeIndex][0];

					// Extract base index and case from neighbor cell(, can be the cell itself too).
					unsigned int uiVal = cGotoZ?pIndexOffsetsZ1[(y-cGotoY)*_pMesh->GetSizeX()+x-cGotoX] : pIndexOffsetsZ0[(y-cGotoY)*_pMesh->GetSizeX()+x-cGotoX];
					unsigned int uiNC = uiVal>>24;
					// Lookup the edge-vertex creations and exculde prior ones
					uiNC = g_acEdges[uiNC] & ((x>cGotoX)?g_uiVertexCreationMaskX:0xfff) & ((y>cGotoY)?g_uiVertexCreationMaskY:0xfff) & ((z>cGotoZ)?g_uiVertexCreationMaskZ:0xfff);
					pIndexData[uiIndex] = (uiVal&0xffffff) + GetEdgeIndexOffset(uiNC, cNeighborEdge);

					++uiIndex;
					if(uiIndex >= (unsigned int)_iTriangles * 3) goto finishcreation;	// break run - to few triangles allocated
				}
			}
		}
		// Going to the next z-layer -> neighbor table is the next one, old one can be overwritten
		unsigned int* pTemp = pIndexOffsetsZ0;
		pIndexOffsetsZ0 = pIndexOffsetsZ1;
		pIndexOffsetsZ1 = pTemp;
	}

finishcreation:
	// Save for statistic and rendercall
	m_iNumIndices = uiIndex;
	m_iNumVertices = uiVertex;

	// Upload to gpu.
	glGenVertexArrays(1, &m_uiVAO);
	glBindVertexArray(m_uiVAO);
	glGenBuffers(1, &m_uiVBO);
    glBindBuffer(GL_ARRAY_BUFFER, m_uiVBO);
	// Upload only necessary data (size of temporary buffer can be much larger than realy used)
	glBufferData(GL_ARRAY_BUFFER, uiVertex * sizeof(SolidVertex), pVertexData, GL_STATIC_DRAW);
	// Insert data and usage declaration
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(SolidVertex), 0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(SolidVertex), (GLvoid*)(3 * sizeof(float)));
	glEnableVertexAttribArray(0);
	// Insert triangulation
	glGenBuffers(1, &m_uiIBO);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_uiIBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * uiIndex, pIndexData, GL_STATIC_DRAW);

	free(pVertexData);
	free(pIndexData);
	free(pIndexOffsetsZ0);
	free(pIndexOffsetsZ1);
}
	
// **************************************************************** //
SolidSurface::~SolidSurface()
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
}

// **************************************************************** //
// Set the buffers and make the rendercall
void SolidSurface::Render()
{
	glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	glBindVertexArray(m_uiVAO);
	//glDrawArrays(GL_POINTS, 0, m_iNumVertices);
	glDrawElements(GL_TRIANGLES, m_iNumIndices, GL_UNSIGNED_INT, (GLvoid*)0);
}