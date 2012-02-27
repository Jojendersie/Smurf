#include <cstdio>
#include <cstring>
#include <cassert>
#include <iostream>
#include "amloader.hpp"
#include "globals.hpp"
//#include <gl>

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

void ToggleEndian(unsigned char* _pcData)
{
	unsigned char ucBuf = _pcData[0]; _pcData[0] = _pcData[3]; _pcData[3] = ucBuf;
	ucBuf = _pcData[1]; _pcData[1] = _pcData[2]; _pcData[2] = ucBuf;
}

// **************************************************************** //
// Functions to find a set of time-slice files
// Returns true if next file name exists and false if all numbers are '9'
bool GoToNextFileName(const char* _pcFileNameMask, char* _pcCurrentName, int _iLen)
{
	bool bOK = false;
	for(int i=_iLen-1; i>=0; --i)
	{
		// Search number mask symbol
		if(_pcFileNameMask[i] == '#')
		{
			// increment the number in the name
			if((_pcCurrentName[i] >= '0') && (_pcCurrentName[i]<'9')) {
				++_pcCurrentName[i];
				return true;
			} else if(_pcCurrentName[i]=='9') {
				// Complex case: carry - loop more, iff loop ends without catching the carry (more numbers in other places) it returns false
				_pcCurrentName[i]='0';
				//bOK = false;
			} else if(_pcCurrentName[i]=='#') {
				_pcCurrentName[i]='0';	// loop more to find other #
				bOK = true;				// If that is the last or only one it is still an valid file name
			}
		}
	}
	return bOK;
}

// **************************************************************** //
// inefficient: testing all filenames to given mask
int CountTimeSlices(const char* _pcFileNameMask, char* _pcCurrentName, int _iLen)
{
	int iCount = 0;
	while(GoToNextFileName(_pcFileNameMask, _pcCurrentName, _iLen))
	{
		FILE* pFile;
		pFile = fopen(_pcCurrentName, "rb");
		if(pFile) // if open succeded then existend
		{
			++iCount;
			fclose(pFile);
		}
	}
	return iCount;
}

// **************************************************************** //
// Load the mesh from file
// Most things copied from http://www.mpi-inf.mpg.de/~weinkauf/notes/amiramesh.html
// Output: Success or not
bool AmiraMesh::Load(const char* _pcFileName)
{
	// Create a working copy of the name, which can be changed.
	int iFNLen = strlen(_pcFileName);
	char acName[256];
	sprintf(acName, "%s", _pcFileName);
	
	// If no time slice masking at least one file will be loaded -> one time slice
	int iTimeSl = CountTimeSlices(_pcFileName, acName, iFNLen);
	m_iSizeT = std::max(1,iTimeSl);
	printf("AmLoader: found %d time slice(s) for %s\n", m_iSizeT, _pcFileName);

	sprintf(acName, "%s", _pcFileName);
	m_pvBuffer = 0;
	for(int i=0; i<m_iSizeT;++i)
	{
		TestNextFile:
		if(iTimeSl && !GoToNextFileName(_pcFileName, acName, iFNLen))
			return false;	// Unexpected error. During counting there were more files
		FILE* pFile;
		pFile = fopen(acName, "rb");
		if(pFile) // if open succeded then load
		{
			if(!_Load(pFile, i)) return false;
			printf("Successfully loaded time slice %d\n", i);
			fclose(pFile);
		} else goto TestNextFile;
	}
	return true;
}

bool AmiraMesh::_Load(FILE* _pFile, int _iSlice)
{
	// We read the first 2k bytes into memory to parse the header.
    // The fixed buffer size looks a bit like a hack, and it is one, but it gets the job done.
    char acBuffer[2048];
    fread(acBuffer, sizeof(char), 2047, _pFile);
    acBuffer[2047] = '\0';		// The following string routines prefer null-terminated strings

	if (!strstr(acBuffer, "# AmiraMesh"))
    {
        printf("Not a proper AmiraMesh file.\n");
        fclose(_pFile);
        return false;
    }

	bool bLittleEndian = strstr(acBuffer, "LITTLE-ENDIAN")!=NULL;

	// Find the Lattice definition, i.e., the dimensions of the uniform grid
	if(3!=sscanf(FindAndJump(acBuffer, "define Lattice"), "%d %d %d\n", &m_iSizeX, &m_iSizeY, &m_iSizeZ))
	{
		printf("'define Lattice' information corrupted.\n");
        fclose(_pFile);
		return false;
	}

	// Find the BoundingBox
	sscanf(FindAndJump(acBuffer, "BoundingBox"), "%g %g %g %g %g %g", &m_vBBMin.x, &m_vBBMax.x, &m_vBBMin.y, &m_vBBMax.y, &m_vBBMin.z, &m_vBBMax.z);

	// Is it a uniform grid? We need this only for the sanity check below.
    if(strstr(acBuffer, "CoordType \"uniform\"") == NULL)
	{
		printf("No uniform coords.\n");
        fclose(_pFile);
		return false;
	}

	// Number of dimensions in one vector (have to be 3 in this application)
    if(!strstr(acBuffer, "Lattice { float[3] Data }"))
    {
        printf("This application uses only 3D-vector fields.\n");
        fclose(_pFile);
		return false;
    }


	// Find the beginning of the data section
    const long idxStartData = 26+strstr(acBuffer, "# Data section follows\n@1\n") - acBuffer;
//	char* pcTest = &acBuffer[idxStartData];
    if (idxStartData > 0)
    {
        // Set the file pointer to the end of "# Data section follows\n@1\n"
        fseek(_pFile, idxStartData, SEEK_SET);

        // Read the data
        // - how much to read
		const size_t NumToRead = m_iSizeX * m_iSizeY * m_iSizeZ;
        // - prepare memory; use malloc() if you're using pure C
		if(!m_pvBuffer) m_pvBuffer = new glm::vec3[NumToRead * m_iSizeT];
        if(!m_pvBuffer)
		{
			printf("Out of memory.\n");
			fclose(_pFile);
			return false;
		}
        
		// - do it
        const size_t ActRead = fread((void*)(m_pvBuffer+NumToRead*_iSlice), sizeof(glm::vec3), NumToRead, _pFile);
        // - ok?
        if (NumToRead != ActRead)
        {
            printf("Something went wrong while reading the binary data section.\nPremature end of file?\n");
            delete[] m_pvBuffer;
            fclose(_pFile);
            return false;
        }


		// Our data have big endian representation!
		if(!bLittleEndian)
		{
			unsigned char* pData = (unsigned char*)(m_pvBuffer+NumToRead*_iSlice);
			for(unsigned int i=0;i<NumToRead*3;++i)
			{
				ToggleEndian(pData);
				pData += 4;
			}
		}

		// Calculate the length of all non solid vectors in respect to the cell size.
		// This value is used to scale the integration step size.
		{
			float fAvgVL = 0.0f;
			glm::vec3* pData = m_pvBuffer+NumToRead*_iSlice;
			int iNumCounted = 0;
			for(unsigned int i=0;i<NumToRead;++i)
			{
				float fLength = glm::length(*pData);
				fAvgVL += fLength;
				if(fLength>0.0f) ++iNumCounted;
				++pData;
			}
			fAvgVL /= iNumCounted;
			m_fAverageVectorLength = (m_fAverageVectorLength*_iSlice + fAvgVL)/(_iSlice+1);
			//float fCellSize = (m_vBBMax.x-m_vBBMin.x)/m_iSizeX;
		}

/*/Test: Print all data values
			float* _pData = (float*)m_pvBuffer;
            //Note: Data runs x-fastest, i.e., the loop over the x-axis is the innermost
            printf("\nPrinting all values in the same order in which they are in memory:\n");
            int Idx(0);
			for(int k=0;k<m_iSizeZ;k++)
            {
				for(int j=0;j<m_iSizeY;j++)
                {
					for(int i=0;i<m_iSizeX;i++)
                    {
                        //Note: Random access to the value (of the first component) of the grid point (i,j,k):
                        // pData[((k * yDim + j) * xDim + i) * NumComponents]
                     //   assert(pData[((k * yDim + j) * xDim + i) * NumComponents] == pData[Idx * NumComponents]);

                        for(int c=0;c<3;c++)
                        {
                            printf("%g ", _pData[Idx * 3 + c]);
                        }
                        printf("\n");
                        Idx++;
                    }
                }
            }
			//*/
	}

	// Optimation stuff
	// Factor m_iSize/(m_vBBMax-m_vBBMin) to translate positions to grid positions faster
	m_vPosToGrid.x = m_iSizeX/(m_vBBMax.x-m_vBBMin.x);
	m_vPosToGrid.y = m_iSizeY/(m_vBBMax.y-m_vBBMin.y);
	m_vPosToGrid.z = m_iSizeZ/(m_vBBMax.z-m_vBBMin.z);

	return true;	// Yes loaded
}


// **************************************************************** //
// Point sampling; Coords have to be in grid space
glm::vec3 AmiraMesh::Sample(float x, float y, float z)
{
	glm::vec3 vOut;
	// Divide into fractal and index
	int ix=(int)x,	iy=(int)y,	iz=(int)z;

	// Edge handling
	if((ix>=m_iSizeX) || (iy>=m_iSizeY) || (iz>=m_iSizeZ) || (ix<0) || (iy<0) || (iz<0))
		return glm::vec3(0.0f);

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

	// Edge handling
	if((ix>=1+m_iSizeX) || (iy>=1+m_iSizeY) || (iz>=1+m_iSizeZ) || (ix<0) || (iy<0) || (iz<0))
		return glm::vec3(0.0f);

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

const float MAXRNDINV = 0.000003051850948f;	// 0.00003051850948f
const float RNDMID = 0.05f;
// **************************************************************** //
// Integrate one step over the vector field to determine new position
// Input:	_vPosition - old position
//			_fStepSize - the size of the integration step; smaller then m_fBBX/m_iSizeX recomended
// Output: new position _fStepSize away from the old one.
glm::vec3 AmiraMesh::Integrate(glm::vec3 _vPosition, float _fStepSize, int _iMethod)
{
	// Translate Position
	glm::vec3 vPos = (_vPosition - m_vBBMin) * m_vPosToGrid;

	// Trilinear sample
	glm::vec3 vS;
	vS = (_iMethod & Globals::INTEGRATION_FILTER_POINT)?Sample(vPos.x,vPos.y,vPos.z):SampleL(vPos.x,vPos.y,vPos.z);

	if(_iMethod & Globals::INTEGRATION_NOISE)
		vS += glm::vec3(rand()*MAXRNDINV-RNDMID, rand()*MAXRNDINV-RNDMID, rand()*MAXRNDINV-RNDMID)*50.0f*m_fAverageVectorLength;

	// Calculate new position
	if(_iMethod & Globals::INTEGRATION_EULER)
		return (vPos + _fStepSize*vS) / m_vPosToGrid + m_vBBMin;
	else
	{
		// Calculate new position with StepSize and with two times
		// StepSize/2. Calculate error from the difference of this two
		// positions. Then extrapolate the real new position.
		glm::vec3 vNewPos1 = vPos + _fStepSize*vS;

		glm::vec3 vNewPos2 = vPos + _fStepSize*0.5f*vS;
		// Resample and step again
		vS = (_iMethod & Globals::INTEGRATION_FILTER_POINT) ?
				  Sample(vNewPos2.x,vNewPos2.y,vNewPos2.z)
				: SampleL(vNewPos2.x,vNewPos2.y,vNewPos2.z);

		vNewPos2 += _fStepSize*0.5f*vS;

		// Extrapolate the position (simple double the difference)
		return (2.0f*vNewPos2-vNewPos1) / m_vPosToGrid + m_vBBMin;

		//vNewPos1 = 2.0f*vNewPos2-vNewPos1;
		// Rescale and return
	//	return glm::vec3(vNewPos1.x/m_fScaleX, );
	}
}

// **************************************************************** //
// Ray casting: if the ray hits the solid this point is returned. Othervise the
// middle point of the straight line through the vectorfield is returned.
// Input:	_vPositions - start of can be inside or in front of the vector field
//			_vDirection - direction of ray
// Output: first solid point in the volume starting at position and shooting into direction.
glm::vec3 AmiraMesh::RayCast(glm::vec3 _vPosition, glm::vec3 _vDirection)
{
	// Transform position to object space
	//_vPosition = (_vPosition - m_vBBMin) * m_vPosToGrid;

	// Calculate entry point to the volume
	// Clipping for each direction. That means, that we move the start point on
	// its ray into the direction of the nearest clipping plane. 
	// (Projection to the supporting plan in ray direction -> finaly the point is:
	//	on the ray, on the plane of the BB, which is hited first from the ray)
	// It is essential, that the projections are computed iteratively, because the
	// coordinates are changing in between.
	// Ray cast from left or right?
	float fDistProjMax = -((_vPosition.x - m_vBBMax.x)/_vDirection.x);	// Search nearest plane
	float fDistProjMin = -((_vPosition.x - m_vBBMin.x)/_vDirection.x);
	if(_vPosition.x < m_vBBMin.x || m_vBBMax.x < _vPosition.x)
		_vPosition += ((fDistProjMin<fDistProjMax)?fDistProjMin:fDistProjMax)*_vDirection;
	// Ray cast from up or down?
	fDistProjMax = -((_vPosition.y - m_vBBMax.y)/_vDirection.y);	// Y
	fDistProjMin = -((_vPosition.y - m_vBBMin.y)/_vDirection.y);
	if(_vPosition.y < m_vBBMin.y || m_vBBMax.y < _vPosition.y)
		_vPosition += ((fDistProjMin<fDistProjMax)?fDistProjMin:fDistProjMax)*_vDirection;
	// Ray cast from front or back?
	fDistProjMax = -((_vPosition.z - m_vBBMax.z)/_vDirection.z);	// Z
	fDistProjMin = -((_vPosition.z - m_vBBMin.z)/_vDirection.z);
	if(_vPosition.z < m_vBBMin.z || m_vBBMax.z < _vPosition.z)
		_vPosition += ((fDistProjMin<fDistProjMax)?fDistProjMin:fDistProjMax)*_vDirection;

	// Transform positions to grid space
	_vPosition = (_vPosition - m_vBBMin) * m_vPosToGrid;
//	vOut = (vOut - m_vBBMin) * m_vPosToGrid;
	_vDirection = glm::normalize(_vDirection * m_vPosToGrid );
//	float fIntersectionLength = glm::length(vOut-_vPosition);

	// Go linear through the volume
	//for(float i=0; i<fIntersectionLength; i+=1.0f)
	glm::vec3 vCurrent = _vPosition;
	while(vCurrent.x<GetSizeX() && vCurrent.y<GetSizeY() && vCurrent.z<GetSizeZ())
	{
		glm::vec3 vSample = Sample(vCurrent.x, vCurrent.y, vCurrent.z);
		// Current point in volume == solid?
		if(abs(vSample.x)+abs(vSample.y)+abs(vSample.z) < 0.00001f)
			// Transform back from grid space
			return (vCurrent-_vDirection)/m_vPosToGrid + m_vBBMin;
		vCurrent += _vDirection;
	}

	// No point was found -> return middle position per default
	return ((vCurrent + _vPosition)*0.5f)/m_vPosToGrid + m_vBBMin;
}