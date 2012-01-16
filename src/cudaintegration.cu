#include <math.h>
#include "cudamath.hpp"

extern "C" float* integrateVectorFieldGPU(float* fVectorField, float* fVertices, float* fDeviceResultVertices, 
										  unsigned int uiElementSize, unsigned int uiBlockSize, int iSizeFieldx, 
										  int iSizeFieldy, int iSizeFieldz, float stepsize, unsigned int bitmask);

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
	float3 Out;

	int3 fi;
	int index;

	fi=convert_int3(Vector);
	index=fi.x+fi.y*Size.x+fi.z*Size.y*Size.x;

	Out.x=Vector_Field[index+0];
	Out.y=Vector_Field[index+1];
	Out.z=Vector_Field[index+2];

	return Out;
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

	Vector-=convert_float3(fi);

	int index;
	for(int i=0;i<8;i++)
	{
		index=fi.x+(i/4) + (fi.y+(i/2))*Size.x + (fi.z+(i/1))*Size.y*Size.x;

		if(index + 2 < Size.x*Size.y*Size.z)
		{
			s[i].x=Vector_Field[index + 0];
			s[i].y=Vector_Field[index + 1];
			s[i].z=Vector_Field[index + 2];
		}
	}

	return lerp(lerp(lerp(s[0],s[4],Vector.x),lerp(s[2],s[6],Vector.x),Vector.y),
				lerp(lerp(s[1],s[5],Vector.x),lerp(s[3],s[7],Vector.x),Vector.y),Vector.z);	
}

__global__ void IntegrateVectorField(const float *Vector_Field, float3 *dptr, int Size_x, int Size_y, int Size_z, float stepsize, unsigned int bitmask)
{
	const int index=blockDim.x*blockIdx.x+threadIdx.x;
	int3 Size;
	Size.x=Size_x;
	Size.y=Size_y;
	Size.z=Size_z;

	float3 clVs,clVertex;

	clVertex=dptr[index];

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

	dptr[index]=clVertex;
}

extern "C" void integrateVectorFieldGPU(float* fVectorField, float3 *dptr, unsigned int uiElementSize, unsigned int uiBlockSize, int iSizeFieldx, int iSizeFieldy, int iSizeFieldz, float stepsize, unsigned int bitmask)
{
	dim3 BlockSize;
	BlockSize.x=uiBlockSize;
	int up=0;
	if(uiElementSize%uiBlockSize!=0)
		up=1;
	dim3 GridSize;
	GridSize.x=(uiElementSize/uiBlockSize)+up;
	
	float *pfTransformedVertices = new float[uiElementSize*3];

	IntegrateVectorField<<<GridSize,BlockSize>>>(fVectorField, dptr,iSizeFieldx,iSizeFieldy,iSizeFieldz,stepsize,bitmask);
	size_t size=uiElementSize*3*sizeof(float);
}
