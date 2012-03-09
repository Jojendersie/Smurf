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
#include <cuda.h>
#include <cuda_runtime_api.h>

texture<float4,3,cudaReadModeElementType> tex;
cudaArray *d_fieldArray;

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
	tmp.x=(float)vec.x;
	tmp.y=(float)vec.y;
	tmp.z=(float)vec.z;
	return tmp;
}

__device__ float3 Sample(float3 Vector, const float *Vector_Field, uint3 Size)
{
	int3 fi;
	int index;

	fi=convert_int3(Vector);
	if(fi.x > Size.x || fi.y > Size.y || fi.z > Size.z || fi.x<0 || fi.y<0 || fi.z<0)
			return make_float3(0,0,0);

	index=fi.x+fi.y*Size.x+fi.z*Size.y*Size.x;

	return make_float3(Vector_Field[(index*3)+0],Vector_Field[(index*3)+1],Vector_Field[(index*3)+2]);

	//float4 tmperg=tex3D(tex,fi.x,fi.y,fi.z);

	//return make_float3(tmperg.x,tmperg.y,tmperg.z);
}

__device__ float3 SampleL(float3 Vector, const float *Vector_Field, uint3 Size)
{
	float3 s[8];

	int3 fi;
	fi=convert_int3(Vector);

	if(fi.x > Size.x || fi.y > Size.y || fi.z > Size.z || fi.x<0 || fi.y<0 || fi.z<0)
			return make_float3(0,0,0);

	Vector=Vector-convert_float3(fi);

	int index;
	for(int i=0;i<8;i++)
	{
		index=fi.x+(i&1) + (fi.y+(i&2))*Size.x + (fi.z+(i&4))*Size.y*Size.x;

		s[i].x=Vector_Field[(index*3) + 0];
		s[i].y=Vector_Field[(index*3) + 1];
		s[i].z=Vector_Field[(index*3) + 2];

		//float4 tmperg=tex3D(tex,fi.x+(i&1),fi.y+(i&2),fi.z+(i&4));

		//s[i]= make_float3(tmperg.x,tmperg.y,tmperg.z);
	}

	return lerp(lerp(lerp(s[0],s[4],Vector.x),lerp(s[2],s[6],Vector.x),Vector.y),
				lerp(lerp(s[1],s[5],Vector.x),lerp(s[3],s[7],Vector.x),Vector.y),Vector.z);	
}

#define MAXRANDINV 0.00000000002328306437f
#define RNDMID 0.028f

__global__ void IntegrateVectorField(float *Vector_Field, float3 *posptr, unsigned int ElementSize, uint3 Size, uint3 rand, float3 bbMin,
									float3 posGridOffset, int resetcolumn, int rows, float stepsize, unsigned int bitmask, float avgVecLength)
{
	const int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index>ElementSize || rows*resetcolumn<index)
		return;
	
	float3 clVs,clVertex;

	clVertex=(posptr[index]-bbMin)*posGridOffset;

	clVs=(bitmask & 0x00000001) ? Sample(clVertex,Vector_Field,Size) : SampleL(clVertex,Vector_Field,Size);

	if(bitmask & 0x00001000)
	{
		float3 rnd;
		rnd.x=random(index+rand.x,(clVs.x+clVs.y+clVs.z)*1000+rand.y);
		rnd.y=random(index+rand.x,(clVs.x+clVs.y+clVs.z)*1000+rand.y);
		rnd.z=random(index+rand.x,(clVs.x+clVs.y+clVs.z)*1000+rand.y);

		clVs+= avgVecLength * make_float3(((rand.z+rnd.x)*MAXRANDINV-RNDMID),((rand.z+rnd.x)*MAXRANDINV-RNDMID),((rand.z+rnd.x)*MAXRANDINV-RNDMID));
	}

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

	posptr[index]=clVertex/posGridOffset+bbMin;
}

__device__ float3 Sample4D(float tInterpolate, unsigned int t0, float3 Vector, const float *Vector_Field, uint4 Size)
{
	int3 fi;
	int index;

	fi=convert_int3(Vector);
	if(fi.x > Size.x || fi.y > Size.y || fi.z > Size.z || fi.x<0 || fi.y<0 || fi.z<0)
		return make_float3(0,0,0);

	index=fi.x+fi.y*Size.x+fi.z*Size.y*Size.x+t0*Size.x*Size.y*Size.z;

	return make_float3(Vector_Field[(index*3)+0],Vector_Field[(index*3)+1],Vector_Field[(index*3)+2]);

	//float4 tmperg=tex3D(tex,fi.x,fi.y,fi.z);

	//return make_float3(tmperg.x,tmperg.y,tmperg.z);
}

__device__ float3 SampleL4D(float tInterpolate, unsigned int t[2], float3 Vector, const float *Vector_Field, uint4 Size)
{
	float3 s[2][8];
	float3 erg[2];

	int3 fi;
	fi=convert_int3(Vector);

	if(fi.x > Size.x || fi.y > Size.y || fi.z > Size.z || fi.x<0 || fi.y<0 || fi.z<0)
		return make_float3(0,0,0);

	Vector=Vector-convert_float3(fi);

	int index;

	for(int j=0;j<2;j++)
	{
		for(int i=0;i<8;i++)
		{
			index=fi.x+(i&1) + (fi.y+(i&2))*Size.x + (fi.z+(i&4))*Size.y*Size.x+t[j]*Size.x*Size.y*Size.z;

			s[j][i].x=Vector_Field[(index*3) + 0];
			s[j][i].y=Vector_Field[(index*3) + 1];
			s[j][i].z=Vector_Field[(index*3) + 2];
			//float4 tmperg=tex3D(tex,fi.x+(i&1),fi.y+(i&2),fi.z+(i&4));

			//s[j][i]= make_float3(tmperg.x,tmperg.y,tmperg.z);
		}

		erg[j] = lerp(lerp(lerp(s[j][0],s[j][4],Vector.x),lerp(s[j][2],s[j][6],Vector.x),Vector.y),
					  lerp(lerp(s[j][1],s[j][5],Vector.x),lerp(s[j][3],s[j][7],Vector.x),Vector.y),Vector.z);	
	}

	return lerp(erg[0],erg[1],tInterpolate);
}

__global__ void IntegrateVectorField4D(float *Vector_Field, float3 *posptr, unsigned int ElementSize, uint4 Size, uint3 rand, float3 bbMin,
									   float3 posGridOffset, int resetcolumn, int rows, float stepsize, unsigned int bitmask, float avgVecLength, uint2 t, float tInterpolate)
{
	const int index=blockDim.x*blockIdx.x+threadIdx.x;
	if(index>ElementSize || rows*resetcolumn<index)
		return;
	
	float3 clVs,clVertex;

	clVertex=(posptr[index]-bbMin)*posGridOffset;

	clVs=(bitmask & 0x00000001) ? Sample4D(tInterpolate,t.x,clVertex,Vector_Field,Size) : SampleL4D(tInterpolate,(unsigned int*)&t,clVertex,Vector_Field,Size);

	if(bitmask & 0x00001000)
	{
		float3 rnd;
		rnd.x=random(index+rand.x,(clVs.x+clVs.y+clVs.z)*1000+rand.y);
		rnd.y=random(index+rand.x,(clVs.x+clVs.y+clVs.z)*1000+rand.y);
		rnd.z=random(index+rand.x,(clVs.x+clVs.y+clVs.z)*1000+rand.y);

		clVs+= avgVecLength * make_float3(((rand.z+rnd.x)*MAXRANDINV-RNDMID),((rand.z+rnd.x)*MAXRANDINV-RNDMID),((rand.z+rnd.x)*MAXRANDINV-RNDMID));
	}

	if(bitmask & 0x00010000)
		clVertex+=(stepsize * clVs);
	else
	{
		float3 clVertexTMP=clVertex+stepsize * clVs;
		clVertex+=(0.5f*stepsize * clVs);
		clVs=(bitmask & 0x00000001) ? Sample4D(tInterpolate,t.x,clVertex,Vector_Field,Size) : SampleL4D(tInterpolate,(unsigned int*)&t,clVertex,Vector_Field,Size);
		clVertex+=(0.5f*stepsize * clVs);
		clVertex=2 * clVertex-clVertexTMP;
	}

	posptr[index]=clVertex/posGridOffset+bbMin;
}

extern "C" void integrateVectorFieldGPU(float* fVectorField, float3 *posptr, unsigned int uiElementSize, unsigned int uiGridSize, 
										unsigned int uiBlockSize, uint4 sizeField, uint3 rnd, float3 bbMin, float3 posGridOff, 
										int resetcolumn, int rows, float stepsize, unsigned int bitmask, float avgVecSize, float tInterpolate, uint2 t)
{
	if(bitmask & 0x00000100)
		IntegrateVectorField4D<<<uiGridSize,uiBlockSize>>>(fVectorField, posptr,uiElementSize,sizeField,rnd,bbMin,posGridOff,resetcolumn,rows,stepsize,bitmask, avgVecSize,t,tInterpolate);
	else
		IntegrateVectorField<<<uiGridSize,uiBlockSize>>>(fVectorField, posptr,uiElementSize,make_uint3(sizeField.x,sizeField.y,sizeField.z),rnd,bbMin,posGridOff,resetcolumn,rows,stepsize,bitmask, avgVecSize);
}

__global__ void ResetColumn(float3* posptr, float3 bbMin, float3 bbMax, int rows, int resetColumn)
{
	const int index=threadIdx.x;
	posptr[resetColumn*rows+index]=lerp(bbMin,bbMax,index/float(rows-1));
}

extern "C" void resetOldColumn(float3* posptr, float3 bbMin, float3 bbMax, int rows, int resetColumn)
{
	ResetColumn<<<1,rows>>>(posptr,bbMin,bbMax,rows,resetColumn);
}

extern "C" void InitCuda(float *vectorField, cudaExtent size)
{
	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float4>();
	cudaMalloc3DArray(&d_fieldArray,&channelDesc,size);

	cudaMemcpy3DParms param = {0};
	param.srcPtr=make_cudaPitchedPtr((void*)vectorField,size.width*sizeof(float)*3,size.width,size.height);
	param.dstArray=d_fieldArray;
	param.extent=size;
	param.kind=cudaMemcpyHostToDevice;
	cudaMemcpy3D(&param);

	tex.normalized=false;
	tex.addressMode[0]=cudaAddressModeWrap;
	tex.addressMode[1]=cudaAddressModeWrap;
	tex.addressMode[2]=cudaAddressModeWrap;

	cudaBindTextureToArray(&tex,d_fieldArray,&channelDesc);
}