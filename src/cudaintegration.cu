////////////////////////////////////////////////////////////////////////////////
//
// Smurf
// =====
// ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
//
// Smoke Surfaces: An Interactive Flow Visualization
// Technique Inspired by Real-World Flow Experiments
//
// File:              /src/cudaintegration.cu
// Author:            Christoph Lämmerhirt
// Creation Date:     2012.01.11
// Description:
//
// Declaration of the interface from C++ to Cuda and the Cuda-Kernel.
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Preprocessor Directives and Namespaces
////////////////////////////////////////////////////////////////////////////////

#include <math.h>
#include "cudamath.hpp"

__device__ int3 convert_int3(float3 vec)
{
	int3 tmp;
	tmp.x=(int)vec.x;
	tmp.y=(int)vec.y;
	tmp.z=(int)vec.z;
	return tmp;
}

__device__ float3 convert_float3(int3 vec)
{
	float3 tmp;
	tmp.x=(int)vec.x;
	tmp.y=(int)vec.y;
	tmp.z=(int)vec.z;
	return tmp;
}

__device__ float3 Sample(float3 Vector, const float *Vector_Field, int3 Size)
{
	int3 fi;
	int index;

	fi=convert_int3(Vector);
	if(fi.x > Size.x || fi.y > Size.y || fi.z > Size.z || fi.x<0 || fi.y<0 || fi.z<0)
			return make_float3(0,0,0);

	index=fi.x+fi.y*Size.x+fi.z*Size.y*Size.x;

	return make_float3(Vector_Field[index+0],Vector_Field[index+1],Vector_Field[index+2]);
}

__device__ float3 lerp(float3 start, float3 end, float t)
{
	return start+t*(end-start);
}

__device__ float3 SampleL(float3 Vector, const float *Vector_Field, int3 Size)
{
	float3 s[8];

	int3 fi;
	fi=convert_int3(Vector);

	if(fi.x > Size.x || fi.y > Size.y || fi.z > Size.z || fi.x<0 || fi.y<0 || fi.z<0)
			return make_float3(0,0,0);

	Vector-=convert_float3(fi);

	int index;
	for(int i=0;i<8;i++)
	{
		index=fi.x+(i/4) + (fi.y+(i/2))*Size.x + (fi.z+(i/1))*Size.y*Size.x;

		s[i].x=Vector_Field[index + 0];
		s[i].y=Vector_Field[index + 1];
		s[i].z=Vector_Field[index + 2];
	}

	return lerp(lerp(lerp(s[0],s[4],Vector.x),lerp(s[2],s[6],Vector.x),Vector.y),
				lerp(lerp(s[1],s[5],Vector.x),lerp(s[3],s[7],Vector.x),Vector.y),Vector.z);	
}

__global__ void IntegrateVectorField(float *Vector_Field, float3 *posptr, float *timeptr, unsigned int ElementSize, unsigned int Size_x, unsigned int Size_y, unsigned int Size_z, float stepsize, unsigned int bitmask)
{
	const int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index>ElementSize)
		return;

	int3 Size;
	Size.x=Size_x;
	Size.y=Size_y;
	Size.z=Size_z;

	float3 clVs,clVertex;

	clVertex=posptr[index];
	timeptr[index]++;//add the amount of time

	clVs=(bitmask & 0x00000001) ? Sample(clVertex,Vector_Field,Size) : SampleL(clVertex,Vector_Field,Size);

	if(bitmask & 0x00010000)
		clVertex+=(stepsize * clVs);
	else
	{
		float3 clVertexTMP=clVertex+stepsize * clVs;
		clVertex+=(0.5f*stepsize * clVs);
		clVs=(bitmask & 0x00000001) ? Sample(clVertex,Vector_Field,Size) : SampleL(clVertex,Vector_Field,Size);
		clVertex+=(0.5f*stepsize * clVs);
		clVertex=2 * clVertex-clVertexTMP;
	}

	posptr[index]=clVertex;
}

extern "C" void integrateVectorFieldGPU(float *fVectorField, float3 *posptr, float *timeptr, unsigned int uiElementSize, unsigned int uiGridSize, unsigned int uiBlockSize, unsigned int iSizeFieldx, unsigned int iSizeFieldy, unsigned int iSizeFieldz, float stepsize, unsigned int bitmask)
{
	dim3 BlockSize;
	BlockSize.x=uiBlockSize;
	dim3 GridSize;
	GridSize.x=uiGridSize;

	IntegrateVectorField<<<GridSize,BlockSize>>>(fVectorField, posptr,timeptr,uiElementSize,iSizeFieldx,iSizeFieldy,iSizeFieldz,stepsize,bitmask);
}
