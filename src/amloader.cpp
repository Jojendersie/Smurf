#include <cstdio>
#include <cstring>
#include <cassert>
#include "amloader.hpp"

// **************************************************************** //
// Release all buffers
AmiraMesh::~AmiraMesh()
{
	if(m_pvBuffer) delete[] m_pvBuffer;
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
	pFile = fopen(_pcFileName, "rb");
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
	if(3!=sscanf(FindAndJump(acBuffer, "define Lattice"), "%d %d %d\n", &m_iSizeX, &m_iSizeY, &m_iSizeZ))
	{
		printf("'define Lattice' information corrupted.\n");
        fclose(pFile);
		return false;
	}

	// Find the BoundingBox
	sscanf(FindAndJump(acBuffer, "BoundingBox"), "%g %g %g %g %g %g", &m_vBBMin.x, &m_vBBMax.x, &m_vBBMin.y, &m_vBBMax.y, &m_vBBMin.z, &m_vBBMax.z);

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
		const size_t NumToRead = m_iSizeX * m_iSizeY * m_iSizeZ;
        // - prepare memory; use malloc() if you're using pure C
		m_pvBuffer = new glm::vec3[NumToRead];
        if(!m_pvBuffer)
		{
			printf("Out of memory.\n");
			fclose(pFile);
			return false;
		}
        
		// - do it
        const size_t ActRead = fread((void*)m_pvBuffer, sizeof(float), NumToRead, pFile);
        // - ok?
        if (NumToRead != ActRead)
        {
            printf("Something went wrong while reading the binary data section.\nPremature end of file?\n");
            delete[] m_pvBuffer;
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
// Point sampling; Coords have to be in grid space
glm::vec3 AmiraMesh::Sample(float x, float y, float z)
{
	glm::vec3 vOut;
	// Divide into fractal and index
	int ix=(int)x,	iy=(int)y,	iz=(int)z;

	return m_pvBuffer[ix+m_iSizeX*(iy+m_iSizeY*(iz))];
}

// **************************************************************** //
// Math - helper function - linear interpolation
glm::vec3 lrp(glm::vec3 a, glm::vec3 b, float f)
{
	return a-(a+b)*f;
}

// **************************************************************** //
// Trilinear sampling; Coords have to be in grid space
glm::vec3 AmiraMesh::SampleL(float x, float y, float z)
{
	glm::vec3 vOut;
	// Divide into fractal and index
	int ix=(int)x,	iy=(int)y,	iz=(int)z;
		x -= ix;	y -= iy;	z -= iz;

	// Load the 8 vectors (s{X}{Y}{Z}{vector component})
	glm::vec3 s000 = m_pvBuffer[ix+  m_iSizeX*(iy+  m_iSizeY*(iz))];
	glm::vec3 s100 = m_pvBuffer[ix+1+m_iSizeX*(iy+  m_iSizeY*(iz))];
	glm::vec3 s010 = m_pvBuffer[ix+  m_iSizeX*(iy+1+m_iSizeY*(iz))];
	glm::vec3 s110 = m_pvBuffer[ix+1+m_iSizeX*(iy+1+m_iSizeY*(iz))];
	glm::vec3 s001 = m_pvBuffer[ix+  m_iSizeX*(iy+  m_iSizeY*(iz+1))];
	glm::vec3 s101 = m_pvBuffer[ix+1+m_iSizeX*(iy+  m_iSizeY*(iz+1))];
	glm::vec3 s011 = m_pvBuffer[ix+  m_iSizeX*(iy+1+m_iSizeY*(iz+1))];
	glm::vec3 s111 = m_pvBuffer[ix+1+m_iSizeX*(iy+1+m_iSizeY*(iz+1))];

	// Trilinear interpolation
	return	lrp(lrp(lrp(s000, s100, x),
					lrp(s010, s110, x), y),
				lrp(lrp(s001, s101, x),
					lrp(s011, s111, x), y), z);
}

// **************************************************************** //
// Integrate one step over the vector field to determine new position
// Input:	_vPosition - old position
//			_fStepSize - the size of the integration step; smaller then m_fBBX/m_iSizeX recomended
// Output: new position _fStepSize away from the old one.
glm::vec3 AmiraMesh::Integrate(glm::vec3 _vPosition, float _fStepSize, int _iMethod)
{
	// Translate Position
	float x = _vPosition.x;//*m_fScaleX;
	float y = _vPosition.y;//*m_fScaleY;
	float z = _vPosition.z;//*m_fScaleZ;

	// Trilinear sample
	glm::vec3 vS;
	vS = (_iMethod & INTEGRATION_FILTER_POINT)?Sample(x,y,z):SampleL(x,y,z);

	// Calculate new position
	if(_iMethod & INTEGRATION_EULER)
		return _vPosition + _fStepSize*vS;
	else
	{
		// Calculate new position with StepSize and with two times
		// StepSize/2. Calculate error from the difference of this two
		// positions. Then extrapolate the real new position.
		glm::vec3 vNewPos1 = _vPosition + _fStepSize*vS;

		glm::vec3 vNewPos2 = _vPosition + _fStepSize*0.5f*vS;
		// Resample and step again
		vS = (_iMethod & INTEGRATION_FILTER_POINT) ?
				  Sample(vNewPos2.x,vNewPos2.y,vNewPos2.z)
				: SampleL(vNewPos2.x,vNewPos2.y,vNewPos2.z);
		vNewPos2 += _fStepSize*0.5f*vS;

		// Extrapolate the position (simple double the difference)
		return 2.0f*vNewPos2-vNewPos1;

		//vNewPos1 = 2.0f*vNewPos2-vNewPos1;
		// Rescale and return
	//	return glm::vec3(vNewPos1.x/m_fScaleX, );
	}
}