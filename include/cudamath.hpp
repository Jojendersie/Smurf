////////////////////////////////////////////////////////////////////////////////
//
// Smurf
// =====
// ##### Martin Kirst, Johannes Jendersie, Christoph Lämmerhirt, Laura Osten #####
//
// Smoke Surfaces: An Interactive Flow Visualization
// Technique Inspired by Real-World Flow Experiments
//
// File:              /include/cudamath.hpp
// Author:            Christoph Lämmerhirt
// Creation Date:     2012.01.11
// Description:
//
// Implementation of some overloaded operators to implement float3 vectors.
//
////////////////////////////////////////////////////////////////////////////////


////////////////////////////////////////////////////////////////////////////////
// Preprocessor Directives and Namespaces
////////////////////////////////////////////////////////////////////////////////

#ifndef CUDAMATH_HPP_
#define CUDAMATH_HPP_

#include <cuda_runtime.h>
#include <vector_types.h>

//////////////////////FLOAT3////////////////////////////////
//NEGATE
inline __host__ __device__ float3 operator -(const float3 &a)
{return make_float3(-a.x,-a.y,-a.z);}

//ADD
inline __host__ __device__ float3 operator +(const float3 &a,const float3 &b)
{ return make_float3(a.x+b.x,a.y+b.y,a.z+b.z); }

inline __host__ __device__ void operator +=(float3 &a,const float3 &b)
{ a.x+=b.x; a.y+=b.y; a.z+=b.z; }

//SUBTRACT
inline __host__ __device__ float3 operator -(const float3 &a,const float3 &b)
{ return make_float3(a.x-b.x,a.y-b.y,a.z-b.z); }

inline __host__ __device__ void operator -=(float3 &a,const float3 &b)
{ a.x-=b.x; a.y-=b.y; a.z-=b.z; }

//MULTIPLY
inline __host__ __device__ float3 operator *(const float3 &a,const float3 &b)
{ return make_float3(a.x*b.x,a.y*b.y,a.z*b.z); }

inline __host__ __device__ float3 operator *(const float3 &a,const float &b)
{ return make_float3(a.x*b,a.y*b,a.z*b); }

inline __host__ __device__ float3 operator *(const float &b, const float3 &a)
{ return make_float3(a.x*b,a.y*b,a.z*b); }

inline __host__ __device__ void operator *=(float3 &a,const float3 &b)
{ a.x*=b.x; a.y*=b.y; a.z*=b.z; }

//DIVIDE
inline __host__ __device__ float3 operator /(const float3 &a,const float3 &b)
{ return make_float3(a.x/b.x,a.y/b.y,a.z/b.z); }

inline __host__ __device__ float3 operator /(const float3 &a,const float &b)
{ return make_float3(a.x/b,a.y/b,a.z/b); }

inline __host__ __device__ float3 operator /(const float &b, const float3 &a)
{ return make_float3(a.x/b,a.y/b,a.z/b); }

inline __host__ __device__ void operator /=(float3 &a,const float3 &b)
{ a.x/=b.x; a.y/=b.y; a.z/=b.z; }


#endif // CUDAMATH_HPP_
