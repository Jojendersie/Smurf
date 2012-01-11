//#include "cuda.h"
//#include "device_launch_parameters.h"

#include "cuda_runtime.h"
#include "vector_types.h"

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