#include "amloader.hpp"
#include <stdio.h>
#include <string.h>
#include <assert.h>

// **************************************************************** //
// Release all buffers
AmiraMesh::~AmiraMesh()
{
	if(m_pfBuffer) delete[] m_pfBuffer;
}

// **************************************************************** //
/** Find a string in the given buffer and return a pointer
    to the contents directly behind the SearchString.
    If not found, return the buffer. A subsequent sscanf()
    will fail then, but at least we return a decent pointer.
*/
const char* FindAndJump(const char* _pcBuffer, const char* _pcSearchString)
{
    const char* pcFoundLoc = strstr(_pcBuffer, _pcSearchString);
    if (pcFoundLoc) return pcFoundLoc + strlen(_pcSearchString);
    return _pcBuffer;
}

// **************************************************************** //
// Load the mesh from file
// Most things copied from http://www.mpi-inf.mpg.de/~weinkauf/notes/amiramesh.html
// Output: Success or not
bool AmiraMesh::Load(char* _pcFileName)
{
	FILE* pFile;
	fopen_s(&pFile, _pcFileName, "rb");
	if(!pFile) return false;	// Cannot open file

	// We read the first 2k bytes into memory to parse the header.
    // The fixed buffer size looks a bit like a hack, and it is one, but it gets the job done.
    char acBuffer[2048];
    fread(acBuffer, sizeof(char), 2047, pFile);
    acBuffer[2047] = '\0';		// The following string routines prefer null-terminated strings

	if (!strstr(acBuffer, "# AmiraMesh"))
    {
        printf("Not a proper AmiraMesh file.\n");
        fclose(pFile);
        return false;
    }

	// Find the Lattice definition, i.e., the dimensions of the uniform grid
	if(3!=sscanf_s(FindAndJump(acBuffer, "define Lattice"), "%d %d %d\n", &m_iSizeX, &m_iSizeY, &m_iSizeZ))
	{
		printf("'define Lattice' information corrupted.\n");
        fclose(pFile);
		return false;
	}

	// Find the BoundingBox
	sscanf_s(FindAndJump(acBuffer, "BoundingBox"), "%g %g %g %g %g %g", &m_vBBMin.x, &m_vBBMax.x, &m_vBBMin.y, &m_vBBMax.y, &m_vBBMin.z, &m_vBBMax.z);

	// Is it a uniform grid? We need this only for the sanity check below.
    if(strstr(acBuffer, "CoordType \"uniform\"") == NULL)
	{
		printf("No uniform coords.\n");
        fclose(pFile);
		return false;
	}

	// Number of dimensions in one vector (have to be 3 in this application)
    if(!strstr(acBuffer, "Lattice { float[3] Data }"))
    {
        printf("This application uses only 3D-vector fields.\n");
        fclose(pFile);
		return false;
    }


	// Find the beginning of the data section
    const long idxStartData = strstr(acBuffer, "# Data section follows\n@1\n") - acBuffer;
    if (idxStartData > 0)
    {
        // Set the file pointer to the end of "# Data section follows\n@1\n"
        fseek(pFile, idxStartData, SEEK_SET);

        // Read the data
        // - how much to read
		const size_t NumToRead = m_iSizeX * m_iSizeY * m_iSizeZ * 3;
        // - prepare memory; use malloc() if you're using pure C
        m_pfBuffer = new float[NumToRead];
        if(!m_pfBuffer)
		{
			printf("Out of memory.\n");
			fclose(pFile);
			return false;
		}
        
		// - do it
        const size_t ActRead = fread((void*)m_pfBuffer, sizeof(float), NumToRead, pFile);
        // - ok?
        if (NumToRead != ActRead)
        {
            printf("Something went wrong while reading the binary data section.\nPremature end of file?\n");
            delete[] m_pfBuffer;
            fclose(pFile);
            return false;
        }
    }

	// Optimation stuff
	// Factor m_iSize/(m_vBBMax-m_vBBMin) to translate positions to grid positions faster
	m_fScaleX = m_iSizeX/(m_vBBMax.x-m_vBBMin.x);
	m_fScaleY = m_iSizeY/(m_vBBMax.y-m_vBBMin.y);
	m_fScaleZ = m_iSizeZ/(m_vBBMax.z-m_vBBMin.z);

	fclose(pFile);
	return true;	// Yes loaded
}


// **************************************************************** //
// Trilinear sampling; Coords have to be in grid space
float3 AmiraMesh::SampleL(float x, float y, float z)
{
	float3 vOut;
	// Divide into fractal and index
	int ix=(int)x,	iy=(int)y,	iz=(int)z;
		x -= ix;	y -= iy;	z -= iz;

	// TODO: Trilinear?
	vOut.x = m_pfBuffer[3*(ix+m_iSizeX*(iy+m_iSizeY*(iz)))  ];
	vOut.y = m_pfBuffer[3*(ix+m_iSizeX*(iy+m_iSizeY*(iz)))+1];
	vOut.z = m_pfBuffer[3*(ix+m_iSizeX*(iy+m_iSizeY*(iz)))+2];

	return vOut;
}

// **************************************************************** //
// Integrate one step over the vector field to determine new position
// Input:	_vPosition - old position
//			_fStepSize - the size of the integration step; smaller then m_fBBX/m_iSizeX recomended
// Output: new position _fStepSize away from the old one.
float3 AmiraMesh::Integrate(float3 _vPosition, float _fStepSize)
{
	// Translate Position
	float x = _vPosition.x*m_fScaleX;
	float y = _vPosition.y*m_fScaleY;
	float z = _vPosition.z*m_fScaleZ;

	// Trilinear sample
	float3 vS = SampleL(x,y,z);

	// Calculate new position
	float3 vOut;
	vOut.x = _vPosition.x + _fStepSize*vS.x;
	vOut.y = _vPosition.y + _fStepSize*vS.y;
	vOut.z = _vPosition.z + _fStepSize*vS.z;
	return vOut;
}