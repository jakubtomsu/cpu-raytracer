
/*
=============================================================================
math library

originally HandmadeMath.h v1.12.1
https://github.com/HandmadeMath/Handmade-Math

This is a single header file with a bunch of useful functions for game and
graphics math operations.

LICENSE
This software is in the public domain. Where that dedication is not
recognized, you are granted a perpetual, irrevocable license to copy,
distribute, and modify this file as you see fit.

CREDITS
Written by Zakary Strange (strangezak@protonmail.com && @strangezak)

Functionality:
	Matt Mascarenhas (@miblo_)
	Aleph
	FieryDrake (@fierydrake)
	Gingerbill (@TheGingerBill)
	Ben Visness (@bvisness)
	Trinton Bullard (@Peliex_Dev)
	@AntonDan

Fixes:
	Jeroen van Rijn (@J_vanRijn)
	Kiljacken (@Kiljacken)
	Insofaras (@insofaras)
	Daniel Gibson (@DanielGibson)
=============================================================================
*/



#ifndef MATHLIB_H_INCLUDED
#define MATHLIB_H_INCLUDED

#include <math.h>
#include <stdlib.h>
#include <float.h>



#ifdef _MSC_VER
	#pragma warning(disable:4201)
#endif

//#if defined(__GNUC__) || defined(__clang__)
//#pragma GCC diagnostic push
//#pragma GCC diagnostic ignored "-Wf32-equal"
//#if defined(__GNUC__) && (__GNUC__ == 4 && __GNUC_MINOR__ < 8)
//#pragma GCC diagnostic ignored "-Wmissing-braces"
//#endif
//#ifdef __clang__
//#pragma GCC diagnostic ignored "-Wgnu-anonymous-struct"
//#endif
//#endif

#ifdef __cplusplus
extern "C" {
#endif



#define M_PI32 3.14159265359f
#define M_PI 3.14159265358979323846
#define M_PI_TIMES_2 6.28318530f


// SIMD
#define MATHLIB_USE_SIMD 1

// steam survey says 98.4% of users have SSE4.2
// AVX2 (256bit simd) has 84% of users
// AVX512 only 2%

//Supported XXX are:
//  Flag    | Arch |  GCC  | Intel CC |  MSVC  |
//----------+------+-------+----------+--------+
// ARM_NEON | ARM  | I & C | None     |   ?    |
// SSE2     | x86  | I & C | I & C    | I & C  |
// SSE3     | x86  | I & C | I & C    | I only |
// SSSE3    | x86  | I & C | I & C    | I only |
// SSE4_1   | x86  | I & C | I & C    | I only |
// SSE4_2   | x86  | I & C | I & C    | I only |
// AVX      | x86  | I & C | I & C    | I & C  |
// AVX2     | x86  | I & C | I & C    | I only |

// mmintrin.h	- MMX - __m64
// xmmintrin.h	- SSE - __m128
// emmintrin.h	- SSE2
// pmmintrin.h	- SSE3
// tmmintrin.h	- SSSE3
// smmintrin.h	- SSE4.1
// nmmintrin.h	- SSE4.2
// ammintrin.h	- SSE4A
// wmmintrin.h	- AES
// immintrin.h	- AVX, AVX2, FMA - __m256
// zmmintrin.h	- AVX 512 - __m512

#if MATHLIB_USE_SIMD
	#ifdef _MSC_VER // MSVC
		#include <intrin.h>
	#else // GCC, clang, icc or something that doesn't support SSE anyway
		#include <immintrin.h>
	#endif
#endif



/*
===============================
math types

type names are inspired by GLSL
===============================
*/

#define MATHLIB_STRUCT_ATTRIB //__attribute__((packed)) // packing should not be necessarry



// 2d 32-bit floating-point vector
// sizeof(vec2f32) == 8 bytes
typedef union MATHLIB_STRUCT_ATTRIB vec2f32 {
	struct {
		float x;
		float y;
	};
	struct {
		float u;
		float v;
	};

	float elements[2];

#if __cplusplus
		float& operator[](const isize i)		{ return elements[i]; }
	const	float& operator[](const isize i) const	{ return elements[i]; }
#endif
} vec2f32, vec2f, vec2;



// 2d 32-bit integer vector
// sizeof(vec2i32) == 8 bytes
typedef union MATHLIB_STRUCT_ATTRIB vec2i32 {
	struct {
		int x;
		int y;
	};
	struct {
		int u;
		int v;
	};

	int elements[2];

#if __cplusplus
		int& operator[](const isize i)		{ return elements[i]; }
	const	int& operator[](const isize i) const	{ return elements[i]; }
#endif
} vec2i32, vec2i;



// 3d 32-bit floating-point vector
// sizeof(vec3f32) == 12 bytes
typedef union MATHLIB_STRUCT_ATTRIB vec3f32 {
	struct {
		float x;
		float y;
		float z;
	};
	struct {
		float u;
		float v;
		float w;
	};
	struct {
		float r;
		float g;
		float b;
	};
	struct {
		vec2f32 xy;
		float ignored0_;
	};
	struct {
		float ignored1_;
		vec2f32 yz;
	};
	struct {
		vec2f32 uv;
		float ignored2_;
	};

	float elements[3];

#if __cplusplus
		float& operator[](const isize i)		{ return elements[i]; }
	const	float& operator[](const isize i) const	{ return elements[i]; }
#endif
} vec3f32, vec3f, vec3;



// 3d 32-bit integer vector
// sizeof(vec3i32) == 12 bytes
typedef union MATHLIB_STRUCT_ATTRIB vec3i32 {
	struct {
		int x;
		int y;
		int z;
	};
	struct {
		int u;
		int v;
		int w;
	};
	struct {
		int r;
		int g;
		int b;
	};
	struct {
		vec2i32 xy;
		int ignored0_;
	};
	struct {
		int ignored1_;
		vec2i32 yz;
	};
	struct {
		vec2i32 uv;
		int ignored2_;
	};

	int elements[3];

#if __cplusplus
		int& operator[](const isize i)		{ return elements[i]; }
	const	int& operator[](const isize i) const	{ return elements[i]; }
#endif
} vec3i32, vec3i;



// 4d 32-bit floating-point vector
// sizeof(vec4f32) == 16
typedef union MATHLIB_STRUCT_ATTRIB vec4f32 {
	struct {
		union {
			vec3f32 xyz;
			struct {
				float x, y, z;
			};
		};
		float w;
	};
	struct {
		union {
			vec3f32 rgb;
			struct {
				float r, g, b;
			};
		};
		float a;
	};
	struct {
		vec2f32 xy;
		float ignored0_;
		float ignored1_;
	};
	struct {
		float ignored2_;
		vec2f32 yz;
		float Ignored3_;
	};
	struct {
		float ignored4_;
		float ignored5_;
		vec2f32 zw;
	};

	float elements[4];

#ifdef MATHLIB_USE_SIMD
	__m128 _elements_sse;
#endif

#if __cplusplus
		float& operator[](const isize i)		{ return elements[i]; }
	const	float& operator[](const isize i) const	{ return elements[i]; }
#endif
} vec4f32, vec4f, vec4;



// 32-bit floating-point quaternion (quaternions are for 3d rotations)
// sizeof(quatf32) == 16 bytes
// basically a vec4f32
typedef union MATHLIB_STRUCT_ATTRIB quatf32 {
	struct {
		union {
			vec3f32 xyz;
			struct {
				float x;
				float y;
				float z;
			};
		};
		float w;
	};

	vec4f32 xyzw;

	float elements[4];

#ifdef MATHLIB_USE_SIMD
	__m128 _elements_sse;
#endif

#if __cplusplus
		float& operator[](const isize i)		{ return elements[i]; }
	const	float& operator[](const isize i) const	{ return elements[i]; }
#endif
} quatf32, quatf, quat;



// 3x3 column-major matrix
// mainly for physics
typedef union MATHLIB_STRUCT_ATTRIB mat3f32 {
	vec3f32 columns[3];
	float elements[3][3];
	struct {
		vec3f32 right;		// x
		vec3f32 up;		// y
		vec3f32 forward;	// z
	};

#if __cplusplus
		vec3f32& operator[](const isize i)		{ return columns[i]; }
	const	vec3f32& operator[](const isize i) const	{ return columns[i]; }
#endif
} mat3f32, mat3f, mat3;



// 4x4 32-bit floating-point column-major matrix
// sizeof(mat4f32) == 64 bytes
typedef union MATHLIB_STRUCT_ATTRIB mat4f32 {
	float	elements[4][4];
	vec4f32	_mat4f32_columns_todo[4];

#ifdef MATHLIB_USE_SIMD
	__m128	columns[4]; // TODO: use vec4f32! pain to rewrite ... :(
	__m128 _elements_sse[4];
#endif

#if __cplusplus
		vec4f32& operator[](const isize i)		{ return _mat4f32_columns_todo[i]; }
	const	vec4f32& operator[](const isize i) const	{ return _mat4f32_columns_todo[i]; }
#endif
} mat4f32, mat4f, mat4;



// 3d axis aligned bounding box
typedef struct MATHLIB_STRUCT_ATTRIB aabb3f32 {
	vec3f32 min;
	vec3f32 max;
} aabb3f32, aabb3f, aabb3;



// 3d transform without scale component
typedef struct MATHLIB_STRUCT_ATTRIB nstransform3f32 {
	vec3f32 position;
	quatf32 rotation;
} nstransform3f32, nstransform3f, nstransform3;



// 3d transform
typedef struct MATHLIB_STRUCT_ATTRIB transform3f32 {
	vec3f32 position;
	quatf32 rotation;
	vec3f32 scale;
} transform3f32, transform3f, transform3;


#if !__cplusplus
typedef unsigned char bool;
#endif



/*
=========
constants
=========
*/

#define F_MAX FLT_MAX
#define F_EPSILON FLT_EPSILON

#define VEC2_ZERO		((vec2){})
#define VEC3_ZERO		((vec3){})
#define VEC4_ZERO		(vec4{})
#define VEC2_ONE		((vec2){1.0f, 1.0f})
#define VEC3_ONE		((vec3){1.0f, 1.0f, 1.0f})
#define VEC4_ONE		(vec4{1.0f, 1.0f, 1.0f, 1.0f})

#define VEC3_COL_WHITE		((vec3){1.0f, 1.0f, 1.0f})
#define VEC3_COL_RED		((vec3){1.0f, 0.0f, 0.0f})
#define VEC3_COL_GREEN		((vec3){1.0f, 0.0f, 0.0f})
#define VEC3_COL_BLUE		((vec3){1.0f, 0.0f, 0.0f})
#define VEC3_COL_YELLOW		((vec3){1.0f, 1.0f, 0.0f})
#define VEC3_COL_MAGNETA	((vec3){1.0f, 0.0f, 1.0f})
#define VEC3_COL_CYAN		((vec3){0.0f, 1.0f, 1.0f})
#define VEC3_COL_BLACK		((vec3){0.0f, 0.0f, 0.0f})

#define QUAT_IDENTITY quat{0.0f, 0.0f, 0.0f, 1.0f}

#define TRANSFORM3_DEFAULT transform3{(vec3){0.0f, 0.0f, 0.0f}, quat{0.0f, 0.0f, 0.0f, 1.0f}, (vec3){1.0f, 1.0f, 1.0f}}
#define NSTRANSFORM3_DEFAULT nstransform3{{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f,1.0f}}

#define VEC3_GPU_COORD_SYSTEM ((vec3){1.0f, 1.0f, -1.0f})
#define MAT4_GPU_COORD_SYSTEM (mat4_scale(VEC3_GPU_COORD_SYSTEM))

#define AABB3_NEGATIVE (aabb3{{INFINITY,INFINITY,INFINITY}, {-INFINITY,-INFINITY,-INFINITY}})


#define MATH_FUNC_DEF static __forceinline

#define GOLDEN_RATIO	1.61803398875f
#define EPSILON		1e-6f
#define SQRT2		1.41421356237f
#define SQRT3		1.73205080757f
#define SQRT2_HALF	0.70710678118f
#ifdef PI
	#error "PI already defined"
#endif
#define PI  3.14159265358f
#define TAU (PI * 2)

#define U16_MAX 0xFFFF
#define I16_MAX 0x7FFF
#define I32_MAX 0x7FFFFFFF





/*
===============
float functions
===============
*/

MATH_FUNC_DEF float f32_abs	(const float a)					{ return(a >= 0.0f ? a : -a); }
MATH_FUNC_DEF float f32_min	(const float a, const float b)			{ return(a < b ? a : b); }
MATH_FUNC_DEF float f32_max	(const float a, const float b)			{ return(a > b ? a : b); }
MATH_FUNC_DEF float f32_clamp	(const float a, const float min, const float max)	{ return(a < min ? min : (a > max ? max : a)); }
MATH_FUNC_DEF int f32_floor	(const float a)					{ return((int)floorf(a)); }
MATH_FUNC_DEF int f32_round	(const float a)					{ return((int)roundf(a)); }
MATH_FUNC_DEF int f32_ceil	(const float a)					{ return((int)ceilf(a)); }
MATH_FUNC_DEF float f32_mod	(const float a, const float b)			{ return(fmod(a, b)); }
MATH_FUNC_DEF float f32_sin	(const float rad)					{ return(sinf	(rad)); }
MATH_FUNC_DEF float f32_cos	(const float rad)					{ return(cosf	(rad)); }
MATH_FUNC_DEF float f32_tan	(const float rad)					{ return(tanf	(rad)); }
MATH_FUNC_DEF float f32_acos	(const float rad)					{ return(acosf	(rad)); }
MATH_FUNC_DEF float f32_atan	(const float rad)					{ return(atanf	(rad));}
MATH_FUNC_DEF float f32_atan2	(const float a, float b)				{ return(atan2f(a, b)); }
MATH_FUNC_DEF float f32_exp	(const float a)					{ return(expf	(a)); }
MATH_FUNC_DEF float f32_log	(const float a)					{ return(logf	(a)); }
MATH_FUNC_DEF float f32_log2	(const float a)					{ return(log2f	(a)); }
MATH_FUNC_DEF float f32_log10	(const float a)					{ return(log10f	(a)); }
MATH_FUNC_DEF float f32_sqare	(const float a)					{ return(a * a); }
MATH_FUNC_DEF float f32_pow	(const float base, const float exponent)		{ return(powf(base, exponent)); }
MATH_FUNC_DEF float f32_to_rad	(const float deg) 				{ return(deg * (PI / 180.0f)); }
MATH_FUNC_DEF float f32_to_deg	(const float rad)					{ return(rad * 57.2957795f); } // radians * (180 / PI)
MATH_FUNC_DEF float f32_lerp	(const float a, const float b, const float time)	{ return((1.0f - time) * a + time * b); }
MATH_FUNC_DEF float f32_roundstep	(const float a, const float step)			{ return((float)f32_round(a / step) * step); }
MATH_FUNC_DEF float f32_sign	(const float a)					{ return(a >= 0.0f ? 1.0f : -1.0f); }
MATH_FUNC_DEF float f32_fract	(const float a)					{ return(a - (float)f32_floor(a)); }

MATH_FUNC_DEF float f32_lerp_clamped(float a, float b, float time) {
	time = f32_clamp(time, 0.0f, 1.0f);
	return(f32_lerp(a, b, time));
}

MATH_FUNC_DEF float f32_sqrt(float a) {
	float result;
#ifdef MATHLIB_USE_SIMD
	__m128 In = _mm_set_ss(a);
	__m128 Out = _mm_sqrt_ss(In);
	result = _mm_cvtss_f32(Out);
#else
	result = sqrtf(a);
#endif
	return(result);
}

MATH_FUNC_DEF float f32_sqrt_inv(float a) {
	float result;
#ifdef MATHLIB_USE_SIMD
	__m128 i = _mm_set_ss(a);
	__m128 o = _mm_rsqrt_ss(i);
	result = _mm_cvtss_f32(o);
#else
	result = 1.0f / f32_sqrt(a);
#endif
	return(result);
}

// same as in OpenGL
MATH_FUNC_DEF float f32_smoothstep(const float edge0, const float edge1, const float x) {
	const float t = f32_clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
	return(t * t * (3.0f - (2.0f * t)));
}

MATH_FUNC_DEF float f32_smoothstep01(const float x) { return f32_smoothstep(0.0f, 1.0f, x); }



/*
=============
int functions
=============
*/

MATH_FUNC_DEF int i32_abs	(const int a)					{ return(a >= 0 ? a : -a); }
MATH_FUNC_DEF int i32_min	(const int a, const int b)			{ return(a < b ? a : b); }
MATH_FUNC_DEF int i32_max	(const int a, const int b)			{ return(a > b ? a : b); }
MATH_FUNC_DEF int i32_clamp	(const int a, const int min, const int max)	{ return(a < min ? min : (a > max ? max : a)); }
MATH_FUNC_DEF int i32_div_ceil	(const int a, const int b)			{ return((a + b - 1) / b); }


MATH_FUNC_DEF int i32_pow(int base, int exponent) {
	if(exponent < 0) { return 0; }

	int result = 1;
	for (;;) {
		if (exponent & 1) { result *= base; }
		exponent >>= 1;
		if (!exponent) { break; }
		base *= base;
	}

	return(result);
}



/*
=============
SSE functions
=============
*/

#ifdef MATHLIB_USE_SIMD

MATH_FUNC_DEF __m128 sse_linear_combine_mat4(__m128 left, mat4 right) {
	__m128 result;
	result = _mm_mul_ps(_mm_shuffle_ps(left, left, 0x00), right.columns[0]);
	result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(left, left, 0x55), right.columns[1]));
	result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(left, left, 0xaa), right.columns[2]));
	result = _mm_add_ps(result, _mm_mul_ps(_mm_shuffle_ps(left, left, 0xff), right.columns[3]));
	return(result);
}

#endif



/*
==============
vec2 functions
==============
*/

// initialize vec2 with integers
MATH_FUNC_DEF vec2 vec2_init_i(int x, int y)	{ return (vec2){(float)x, (float)y}; }
MATH_FUNC_DEF vec2 vec2_init_f(float f)		{ return((vec2){f, f}); }
MATH_FUNC_DEF vec2 vec2_negate(vec2 a)		{ return((vec2){-a.x, -a.y}); }

MATH_FUNC_DEF vec2 vec2_add(vec2 left, vec2 right) {
	vec2 result;
	result.x = left.x + right.x;
	result.y = left.y + right.y;
	return(result);
}

MATH_FUNC_DEF vec2 vec2_sub(vec2 left, vec2 right) {
	vec2 result;
	result.x = left.x - right.x;
	result.y = left.y - right.y;
	return(result);
}

MATH_FUNC_DEF vec2 vec2_mul(vec2 left, vec2 right) {
	vec2 result;
	result.x = left.x * right.x;
	result.y = left.y * right.y;
	return(result);
}

MATH_FUNC_DEF vec2 vec2_mul_f(vec2 left, float right) {
	vec2 result;
	result.x = left.x * right;
	result.y = left.y * right;
	return(result);
}

MATH_FUNC_DEF vec2 vec2_div(vec2 left, vec2 right) {
	vec2 result;
	result.x = left.x / right.x;
	result.y = left.y / right.y;
	return(result);
}

MATH_FUNC_DEF vec2 vec2_div_f(vec2 left, float right) {
	vec2 result;
	result.x = left.x / right;
	result.y = left.y / right;
	return(result);
}

MATH_FUNC_DEF bool vec2_equals(vec2 left, vec2 right)	{ return(left.x == right.x && left.y == right.y); }
MATH_FUNC_DEF float vec2_dot(vec2 a, vec2 b)		{ return(a.x * b.x) + (a.y * b.y); }
MATH_FUNC_DEF float vec2_len_sq(vec2 a)			{ return(vec2_dot(a, a)); }
MATH_FUNC_DEF float vec2_len(vec2 a)			{ return(f32_sqrt(vec2_len_sq(a))); }
// is faster, but less accurate
MATH_FUNC_DEF vec2 vec2_fast_normalize(vec2 a)		{ return(vec2_mul_f(a, f32_sqrt_inv(vec2_dot(a, a)))); }

MATH_FUNC_DEF vec2 vec2_normalize(vec2 a) {
	vec2 result = {0};
	float len = vec2_len(a);
	/* NOTE(kiljacken): We need a zero check to not divide-by-zero */
	if(len != 0.0f) {
		result.x = a.x * (1.0f / len);
		result.y = a.y * (1.0f / len);
	}

	return(result);
}



/*
==============
vec3 functions
==============
*/

MATH_FUNC_DEF vec3 vec3_init_i(int x, int y, int z)	{ return((vec3){ (float)x, (float)y, (float)z }); }
MATH_FUNC_DEF vec3 vec3_init_f(float f)			{ return((vec3){f, f, f}); }
MATH_FUNC_DEF vec3 vec3_negate(vec3 v)			{ return((vec3){-v.x, -v.y, -v.z}); }

MATH_FUNC_DEF vec3 vec3_add(vec3 left, vec3 right) {
	vec3 result;
	result.x = left.x + right.x;
	result.y = left.y + right.y;
	result.z = left.z + right.z;
	return(result);
}

MATH_FUNC_DEF vec3 vec3_sub(vec3 left, vec3 right) {
	vec3 result;
	result.x = left.x - right.x;
	result.y = left.y - right.y;
	result.z = left.z - right.z;
	return(result);
}

MATH_FUNC_DEF vec3 vec3_mul(vec3 left, vec3 right) {
	vec3 result;
	result.x = left.x * right.x;
	result.y = left.y * right.y;
	result.z = left.z * right.z;
	return(result);
}

MATH_FUNC_DEF vec3 vec3_mul_f(vec3 left, float right) {
	vec3 result;
	result.x = left.x * right;
	result.y = left.y * right;
	result.z = left.z * right;
	return(result);
}

MATH_FUNC_DEF vec3 vec3_div(vec3 left, vec3 right) {
	vec3 result;
	result.x = left.x / right.x;
	result.y = left.y / right.y;
	result.z = left.z / right.z;
	return(result);
}

MATH_FUNC_DEF vec3 vec3_div_f(vec3 left, float right) {
	vec3 result;
	result.x = left.x / right;
	result.y = left.y / right;
	result.z = left.z / right;
	return(result);
}

MATH_FUNC_DEF bool vec3_equals(vec3 left, vec3 right) { return  (left.x == right.x && left.y == right.y && left.z == right.z); }
MATH_FUNC_DEF vec3 vec3_abs(const vec3 a) { return((vec3){f32_abs(a.x), f32_abs(a.y), f32_abs(a.z)}); }
MATH_FUNC_DEF vec3 vec3_min(const vec3 a, const vec3 b) { return((vec3){f32_min(a.x, b.x), f32_min(a.y, b.y), f32_min(a.z, b.z)}); }
MATH_FUNC_DEF vec3 vec3_max(const vec3 a, const vec3 b) { return((vec3){f32_max(a.x, b.x), f32_max(a.y, b.y), f32_max(a.z, b.z)}); }
MATH_FUNC_DEF float vec3_dot(vec3 VecOne, vec3 VecTwo) { return(VecOne.x * VecTwo.x) + (VecOne.y * VecTwo.y) + (VecOne.z * VecTwo.z); }

MATH_FUNC_DEF int vec3_min_index(const vec3 v) {
	if(v.x < v.y) {
		if(v.x < v.z) { return(0); }
		return(2);
	}
	if(v.y < v.z) { return(1); }
	return(2);
}


MATH_FUNC_DEF int vec3_max_index(const vec3 v) {
	if(v.x > v.y) {
		if(v.x > v.z) { return(0); }
		return(2);
	}
	if(v.y > v.z) { return(1); }
	return(2);
}


MATH_FUNC_DEF vec3 vec3_cross(vec3 v1, vec3 v2) {
	return((vec3){(v1.y * v2.z) - (v1.z * v2.y), (v1.z * v2.x) - (v1.x * v2.z), (v1.x * v2.y) - (v1.y * v2.x)});
}

MATH_FUNC_DEF float vec3_len_sq(vec3 A)	{ return(vec3_dot(A, A)); }
MATH_FUNC_DEF float vec3_len(vec3 A)		{ return(f32_sqrt(vec3_len_sq(A))); }
MATH_FUNC_DEF vec3 vec3_fast_normalize(vec3 A)	{ return(vec3_mul_f(A, f32_sqrt_inv(vec3_dot(A, A)))); }

MATH_FUNC_DEF vec3 vec3_normalize(vec3 A) {
	vec3 result = {0};
	float VectorLength = vec3_len(A);
	/* NOTE(kiljacken): We need a zero check to not divide-by-zero */
	if(VectorLength != 0.0f) {
		result.x = A.x * (1.0f / VectorLength);
		result.y = A.y * (1.0f / VectorLength);
		result.z = A.z * (1.0f / VectorLength);
	}

	return(result);
}

MATH_FUNC_DEF vec3 vec3_lerp(vec3 A, vec3 B, float Time) {
	return(vec3_add(vec3_mul_f(A, (1.0f - Time)), vec3_mul_f(B, Time)));
}

MATH_FUNC_DEF vec3 vec3_nlerp(const vec3 a, const vec3 b, const float t) {
	const float a_len = vec3_len(a);
	const float b_len = vec3_len(b);
	if(a_len == 0 || b_len == 0) { return(vec3_lerp(a, b, t)); }
	return(vec3_mul_f(vec3_lerp(vec3_div_f(a, a_len), vec3_div_f(b, b_len), t), f32_lerp(a_len, b_len, t)));
}

MATH_FUNC_DEF vec3 vec3_reflect(vec3 V, vec3 N) {
	float dn = 2 * vec3_dot(V, N);
	return vec3_sub(V, vec3_mul_f(N, dn));
}

MATH_FUNC_DEF vec3 quat_mul_vec3(vec3 v, quat q) {
	vec3 t = vec3_mul_f(vec3_cross(q.xyz, v), 2.0f);
	vec3 result = vec3_add(vec3_add(v, vec3_mul_f(t, q.w)), vec3_cross(q.xyz, t));
	return(result);
}

MATH_FUNC_DEF vec3 vec3_to_rad(vec3 v) {
	vec3 result = {
		f32_to_rad(v.x),
		f32_to_rad(v.y),
		f32_to_rad(v.z)
	};
	return(result);
}

MATH_FUNC_DEF vec3 vec3_to_deg(vec3 v) {
	vec3 result = {
		f32_to_deg(v.x),
		f32_to_deg(v.y),
		f32_to_deg(v.z)
	};
	return(result);
}

MATH_FUNC_DEF float vec3_angle(vec3 a, vec3 b) {
	return(f32_acos(vec3_dot(a, b) / (vec3_len(a) * vec3_len(b))));
}

MATH_FUNC_DEF bool vec3_isfinite(vec3 v) {
	v = vec3_abs(v);
	bool result = (
		v.x != INFINITY &&
		v.y != INFINITY &&
		v.z != INFINITY &&
		!isnan(v.x) &&
		!isnan(v.x) &&
		!isnan(v.x)
	);

	return(result);
}

MATH_FUNC_DEF vec3 vec3_hsv_to_rgb(vec3 hsv) {
	const vec4 k = (vec4){1.0f, 2.0f/3.0f, 1.0f/3.0f, 3.0};
	const vec3 fract = (vec3){f32_fract(hsv.x+k.x), f32_fract(hsv.x+k.y), f32_fract(hsv.x+k.z)};
	const vec3 p = vec3_abs(vec3_sub(vec3_mul_f(fract, 6.0f), (vec3){k.w, k.w, k.w}));
	return(vec3_mul_f(vec3_lerp((vec3){k.x, k.x, k.x},
		(vec3){f32_clamp(p.x - k.x, 0.0, 1.0), f32_clamp(p.y - k.x, 0.0, 1.0), f32_clamp(p.z - k.x, 0.0, 1.0)}, hsv.y), hsv.z));
}

MATH_FUNC_DEF float vec3_rgb_to_bw(const vec3 rgb) {
	const vec3 factors = (vec3){0.2126f, 0.7152f, 0.0722f};
	return vec3_dot(rgb, factors);
}

MATH_FUNC_DEF vec3 vec3_to_gpu_coord_sys(const vec3 v) { return (vec3){v.x, v.y, -v.z}; }


/*
==============
vec4 functions
==============
*/

MATH_FUNC_DEF vec4 vec4_add(vec4 left, vec4 right) {
	vec4 result;
#ifdef MATHLIB_USE_SIMD
	result._elements_sse = _mm_add_ps(left._elements_sse, right._elements_sse);
#else
	result.x = left.x + right.x;
	result.y = left.y + right.y;
	result.z = left.z + right.z;
	result.w = left.w + right.w;
#endif
	return(result);
}

MATH_FUNC_DEF vec4 vec4_i(int X, int Y, int Z, int W) {
	vec4 result;
#ifdef MATHLIB_USE_SIMD
	result._elements_sse = _mm_setr_ps((float)X, (float)Y, (float)Z, (float)W);
#else
	result.x = (float)X;
	result.y = (float)Y;
	result.z = (float)Z;
	result.w = (float)W;
#endif
	return(result);
}

MATH_FUNC_DEF vec4 vec4_sub(vec4 left, vec4 right) {
	vec4 result;
#ifdef MATHLIB_USE_SIMD
	result._elements_sse = _mm_sub_ps(left._elements_sse, right._elements_sse);
#else
	result.x = left.x - right.x;
	result.y = left.y - right.y;
	result.z = left.z - right.z;
	result.w = left.w - right.w;
#endif
	return(result);
}


MATH_FUNC_DEF vec4 vec4_mul(vec4 left, vec4 right) {
	vec4 result;
#ifdef MATHLIB_USE_SIMD
	result._elements_sse = _mm_mul_ps(left._elements_sse, right._elements_sse);
#else
	result.x = left.x * right.x;
	result.y = left.y * right.y;
	result.z = left.z * right.z;
	result.w = left.w * right.w;
#endif
	return(result);
}

MATH_FUNC_DEF vec4 vec4_mul_f(vec4 left, float right) {
	vec4 result;
#ifdef MATHLIB_USE_SIMD
	__m128 Scalar = _mm_set1_ps(right);
	result._elements_sse = _mm_mul_ps(left._elements_sse, Scalar);
#else
	result.x = left.x * right;
	result.y = left.y * right;
	result.z = left.z * right;
	result.w = left.w * right;
#endif
	return(result);
}

MATH_FUNC_DEF vec4 vec4_div(vec4 left, vec4 right) {
	vec4 result;
#ifdef MATHLIB_USE_SIMD
	result._elements_sse = _mm_div_ps(left._elements_sse, right._elements_sse);
#else
	result.x = left.x / right.x;
	result.y = left.y / right.y;
	result.z = left.z / right.z;
	result.w = left.w / right.w;
#endif
	return(result);
}

MATH_FUNC_DEF vec4 vec4_div_f(vec4 left, float right) {
	vec4 result;
#ifdef MATHLIB_USE_SIMD
	__m128 Scalar = _mm_set1_ps(right);
	result._elements_sse = _mm_div_ps(left._elements_sse, Scalar);
#else
	result.x = left.x / right;
	result.y = left.y / right;
	result.z = left.z / right;
	result.w = left.w / right;
#endif
	return(result);
}

MATH_FUNC_DEF bool vec4_equals(vec4 left, vec4 right) {
	return(left.x == right.x && left.y == right.y && left.z == right.z && left.w == right.w);
}

MATH_FUNC_DEF float vec4_dot(vec4 VecOne, vec4 VecTwo) {
	float result;
	// NOTE(zak): IN the future if we wanna check what version SSE is support
	// we can use _mm_dp_ps (4.3) but for now we will use the old way.
	// Or a r = _mm_mul_ps(v1, v2), r = _mm_hadd_ps(r, r), r = _mm_hadd_ps(r, r) for SSE3
#ifdef MATHLIB_USE_SIMD
	__m128 SSEResultOne = _mm_mul_ps(VecOne._elements_sse, VecTwo._elements_sse);
	__m128 SSEResultTwo = _mm_shuffle_ps(SSEResultOne, SSEResultOne, _MM_SHUFFLE(2, 3, 0, 1));
	SSEResultOne = _mm_add_ps(SSEResultOne, SSEResultTwo);
	SSEResultTwo = _mm_shuffle_ps(SSEResultOne, SSEResultOne, _MM_SHUFFLE(0, 1, 2, 3));
	SSEResultOne = _mm_add_ps(SSEResultOne, SSEResultTwo);
	_mm_store_ss(&result, SSEResultOne);
#else
	result = (VecOne.x * VecTwo.x) + (VecOne.y * VecTwo.y) + (VecOne.z * VecTwo.z) + (VecOne.w * VecTwo.w);
#endif

	return(result);
}

MATH_FUNC_DEF float vec4_length_sq(vec4 a)	{ return(vec4_dot(a, a)); }
MATH_FUNC_DEF float vec4_len(vec4 a)			{ return(f32_sqrt(vec4_length_sq(a))); }
// faster, but less accurate
MATH_FUNC_DEF vec4 vec4_fast_normalize(vec4 A) { return(vec4_mul_f(A, f32_sqrt_inv(vec4_dot(A, A)))); }

MATH_FUNC_DEF vec4 vec4_normalize(vec4 A) {
	vec4 result = {0};
	float VectorLength = vec4_len(A);
	// NOTE(kiljacken): We need a zero check to not divide-by-zero
	if(VectorLength != 0.0f) {
		float Multiplier = 1.0f / VectorLength;

#ifdef MATHLIB_USE_SIMD
		__m128 SSEMultiplier = _mm_set1_ps(Multiplier);
		result._elements_sse = _mm_mul_ps(A._elements_sse, SSEMultiplier);
#else
		result.x = A.x * Multiplier;
		result.y = A.y * Multiplier;
		result.z = A.z * Multiplier;
		result.w = A.w * Multiplier;
#endif
	}

	return(result);
}

MATH_FUNC_DEF vec4 vec4_lerp(const vec4 start, const vec4 end, const float time) {
	return(vec4_add(vec4_mul_f(start, (1.0f - time)), vec4_mul_f(end, time)));
}



#ifdef __cplusplus
}
#endif


#if defined(__GNUC__) || defined(__clang__)
//#pragma GCC diagnostic pop
#endif



/*
==============
quat functions
==============
*/

MATH_FUNC_DEF quat quat_add(quat left, quat right) {
	quat result;

#ifdef MATHLIB_USE_SIMD
	result._elements_sse = _mm_add_ps(left._elements_sse, right._elements_sse);
#else
	result.x = left.x + right.x;
	result.y = left.y + right.y;
	result.z = left.z + right.z;
	result.w = left.w + right.w;
#endif

	return(result);
}

MATH_FUNC_DEF quat quat_sub(quat left, quat right) {
	quat result;

#ifdef MATHLIB_USE_SIMD
	result._elements_sse = _mm_sub_ps(left._elements_sse, right._elements_sse);
#else
	result.x = left.x - right.x;
	result.y = left.y - right.y;
	result.z = left.z - right.z;
	result.w = left.w - right.w;
#endif

	return(result);
}

MATH_FUNC_DEF quat quat_mul(quat left, quat right) {
	quat result;

#ifdef MATHLIB_USE_SIMD
	__m128 SSEResultOne = _mm_xor_ps(_mm_shuffle_ps(left._elements_sse, left._elements_sse, _MM_SHUFFLE(0, 0, 0, 0)),_mm_setr_ps(0.f, -0.f, 0.f, -0.f));
	__m128 SSEResultTwo = _mm_shuffle_ps(right._elements_sse, right._elements_sse, _MM_SHUFFLE(0, 1, 2, 3));
	__m128 SSEResultThree = _mm_mul_ps(SSEResultTwo, SSEResultOne);

	SSEResultOne = _mm_xor_ps(_mm_shuffle_ps(left._elements_sse, left._elements_sse, _MM_SHUFFLE(1, 1, 1, 1)),_mm_setr_ps(0.f, 0.f, -0.f, -0.f));
	SSEResultTwo = _mm_shuffle_ps(right._elements_sse, right._elements_sse, _MM_SHUFFLE(1, 0, 3, 2));
	SSEResultThree = _mm_add_ps(SSEResultThree, _mm_mul_ps(SSEResultTwo, SSEResultOne));

	SSEResultOne = _mm_xor_ps(_mm_shuffle_ps(left._elements_sse, left._elements_sse, _MM_SHUFFLE(2, 2, 2, 2)), _mm_setr_ps(0.f, 0.f, 0.f, -0.f));
	SSEResultTwo = _mm_shuffle_ps(right._elements_sse, right._elements_sse, _MM_SHUFFLE(2, 3, 0, 1));
	SSEResultThree = _mm_add_ps(SSEResultThree, _mm_mul_ps(SSEResultTwo, SSEResultOne));

	SSEResultOne = _mm_shuffle_ps(left._elements_sse, left._elements_sse, _MM_SHUFFLE(3, 3, 3, 3));
	SSEResultTwo = _mm_shuffle_ps(right._elements_sse, right._elements_sse, _MM_SHUFFLE(3, 2, 1, 0));
	result._elements_sse = _mm_add_ps(SSEResultThree, _mm_mul_ps(SSEResultTwo, SSEResultOne));
	#else
	result.x = (left.x * right.w) + (left.y * right.z) - (left.z * right.y) + (left.w * right.x);
	result.y = (-left.x * right.z) + (left.y * right.w) + (left.z * right.x) + (left.w * right.y);
	result.z = (left.x * right.y) - (left.y * right.x) + (left.z * right.w) + (left.w * right.z);
	result.w = (-left.x * right.x) - (left.y * right.y) - (left.z * right.z) + (left.w * right.w);
#endif

	return(result);
}

MATH_FUNC_DEF quat quat_mul_f(quat left, float Multiplicative) {
	quat result;

#ifdef MATHLIB_USE_SIMD
	__m128 Scalar = _mm_set1_ps(Multiplicative);
	result._elements_sse = _mm_mul_ps(left._elements_sse, Scalar);
#else
	result.x = left.x * Multiplicative;
	result.y = left.y * Multiplicative;
	result.z = left.z * Multiplicative;
	result.w = left.w * Multiplicative;
#endif

	return(result);
}

MATH_FUNC_DEF quat quat_div_f(quat left, float Dividend) {
	quat result;

#ifdef MATHLIB_USE_SIMD
	__m128 Scalar = _mm_set1_ps(Dividend);
	result._elements_sse = _mm_div_ps(left._elements_sse, Scalar);
#else
	result.x = left.x / Dividend;
	result.y = left.y / Dividend;
	result.z = left.z / Dividend;
	result.w = left.w / Dividend;
#endif

	return(result);
}

MATH_FUNC_DEF float quat_dot(quat left, quat right) {
	float result;

#ifdef MATHLIB_USE_SIMD
	__m128 SSEResultOne = _mm_mul_ps(left._elements_sse, right._elements_sse);
	__m128 SSEResultTwo = _mm_shuffle_ps(SSEResultOne, SSEResultOne, _MM_SHUFFLE(2, 3, 0, 1));
	SSEResultOne = _mm_add_ps(SSEResultOne, SSEResultTwo);
	SSEResultTwo = _mm_shuffle_ps(SSEResultOne, SSEResultOne, _MM_SHUFFLE(0, 1, 2, 3));
	SSEResultOne = _mm_add_ps(SSEResultOne, SSEResultTwo);
	_mm_store_ss(&result, SSEResultOne);
#else
	result = (left.x * right.x) + (left.y * right.y) + (left.z * right.z) + (left.w * right.w);
#endif

	return(result);
}

MATH_FUNC_DEF float quat_len_sq(const quat a) { return quat_dot(a, a); }
MATH_FUNC_DEF float quat_len(const quat a) { return f32_sqrt(quat_len_sq(a)); }

MATH_FUNC_DEF quat quat_normalize(quat a) {
	quat result;
	const float length = quat_len(a);
	result = quat_div_f(a, length);
	return(result);
}



/*
==================
operator functions
==================
*/

#ifdef __cplusplus

MATH_FUNC_DEF vec2 operator+(const vec2 left, const vec2 right) { return(vec2_add	(left, right)); }
MATH_FUNC_DEF vec3 operator+(const vec3 left, const vec3 right) { return(vec3_add	(left, right)); }
MATH_FUNC_DEF vec4 operator+(const vec4 left, const vec4 right) { return(vec4_add	(left, right)); }
MATH_FUNC_DEF quat operator+(const quat left, const quat right) { return(quat_add	(left, right)); }

MATH_FUNC_DEF vec2 operator-(const vec2 left, const vec2 right) { return(vec2_sub	(left, right)); }
MATH_FUNC_DEF vec3 operator-(const vec3 left, const vec3 right) { return(vec3_sub	(left, right)); }
MATH_FUNC_DEF vec4 operator-(const vec4 left, const vec4 right) { return(vec4_sub	(left, right)); }

MATH_FUNC_DEF vec2 operator*(const vec2		left, const vec2	right) { return(vec2_mul	(left,	right)); }
MATH_FUNC_DEF vec3 operator*(const vec3		left, const vec3	right) { return(vec3_mul	(left,	right)); }
MATH_FUNC_DEF vec4 operator*(const vec4		left, const vec4	right) { return(vec4_mul	(left,	right)); }
MATH_FUNC_DEF quat operator*(const quat		left, const quat	right) { return(quat_mul	(left,	right)); }
MATH_FUNC_DEF vec2 operator*(const vec2		left, const float		right) { return(vec2_mul_f	(left,	right)); }
MATH_FUNC_DEF vec3 operator*(const vec3		left, const float		right) { return(vec3_mul_f	(left,	right)); }
MATH_FUNC_DEF vec4 operator*(const vec4		left, const float		right) { return(vec4_mul_f	(left,	right)); }
MATH_FUNC_DEF quat operator*(const quat		left, const float		right) { return(quat_mul_f	(left,	right)); }
MATH_FUNC_DEF vec2 operator*(const float		left, const vec2	right) { return(vec2_mul_f	(right,	left)); }
MATH_FUNC_DEF vec3 operator*(const float		left, const vec3	right) { return(vec3_mul_f	(right,	left)); }
MATH_FUNC_DEF vec4 operator*(const float		left, const vec4	right) { return(vec4_mul_f	(right,	left)); }
MATH_FUNC_DEF quat operator*(const float		left, const quat	right) { return(quat_mul_f	(right,	left)); }
MATH_FUNC_DEF vec3 operator*(const quat		left, const vec3	right) { return(quat_mul_vec3	(right,	left)); }
MATH_FUNC_DEF vec3 operator*(const vec3		left, const quat	right) { return(quat_mul_vec3	(left,	right)); }

MATH_FUNC_DEF vec2 operator/(const vec2 left, const vec2	right) { return(vec2_div	(left, right)); }
MATH_FUNC_DEF vec3 operator/(const vec3 left, const vec3	right) { return(vec3_div	(left, right)); }
MATH_FUNC_DEF vec4 operator/(const vec4 left, const vec4	right) { return(vec4_div	(left, right)); }
MATH_FUNC_DEF vec2 operator/(const vec2 left, const float		right) { return(vec2_div_f	(left, right)); }
MATH_FUNC_DEF vec3 operator/(const vec3 left, const float		right) { return(vec3_div_f	(left, right)); }
MATH_FUNC_DEF vec4 operator/(const vec4 left, const float		right) { return(vec4_div_f	(left, right)); }
MATH_FUNC_DEF quat operator/(const quat left, const float		right) { return(quat_div_f	(left, right)); }



MATH_FUNC_DEF vec2& operator+=(vec2& left, const vec2 right) { return(left = vec2_add(left, right)); }
MATH_FUNC_DEF vec3& operator+=(vec3& left, const vec3 right) { return(left = vec3_add(left, right)); }
MATH_FUNC_DEF vec4& operator+=(vec4& left, const vec4 right) { return(left = vec4_add(left, right)); }
MATH_FUNC_DEF quat& operator+=(quat& left, const quat right) { return(left = quat_add(left, right)); }

MATH_FUNC_DEF vec2& operator-=(vec2& left, const vec2 right) { return(left = vec2_sub(left, right)); }
MATH_FUNC_DEF vec3& operator-=(vec3& left, const vec3 right) { return(left = vec3_sub(left, right)); }
MATH_FUNC_DEF vec4& operator-=(vec4& left, const vec4 right) { return(left = vec4_sub(left, right)); }
MATH_FUNC_DEF quat& operator-=(quat& left, const quat right) { return(left = quat_sub(left, right)); }

MATH_FUNC_DEF vec2& operator*=(vec2& left, const vec2	right) { return(left = vec2_mul		(left, right)); }
MATH_FUNC_DEF vec3& operator*=(vec3& left, const vec3	right) { return(left = vec3_mul		(left, right)); }
MATH_FUNC_DEF vec4& operator*=(vec4& left, const vec4	right) { return(left = vec4_mul		(left, right)); }
MATH_FUNC_DEF vec2& operator*=(vec2& left, const float	right) { return(left = vec2_mul_f	(left, right)); }
MATH_FUNC_DEF vec3& operator*=(vec3& left, const float	right) { return(left = vec3_mul_f	(left, right)); }
MATH_FUNC_DEF vec4& operator*=(vec4& left, const float	right) { return(left = vec4_mul_f	(left, right)); }
MATH_FUNC_DEF quat& operator*=(quat& left, const float	right) { return(left = quat_mul_f	(left, right)); }
MATH_FUNC_DEF quat& operator*=(quat& left, const quat	right) { return(left = quat_mul		(left, right)); }
MATH_FUNC_DEF vec3& operator*=(vec3& left, const quat	right) { return(left = quat_mul_vec3	(left, right)); }

MATH_FUNC_DEF vec2& operator/=(vec2& left, const vec2	right) { return(left = vec2_div		(left, right)); }
MATH_FUNC_DEF vec3& operator/=(vec3& left, const vec3	right) { return(left = vec3_div		(left, right)); }
MATH_FUNC_DEF vec4& operator/=(vec4& left, const vec4	right) { return(left = vec4_div		(left, right)); }
MATH_FUNC_DEF vec2& operator/=(vec2& left, const float	right) { return(left = vec2_div_f	(left, right)); }
MATH_FUNC_DEF vec3& operator/=(vec3& left, const float	right) { return(left = vec3_div_f	(left, right)); }
MATH_FUNC_DEF vec4& operator/=(vec4& left, const float	right) { return(left = vec4_div_f	(left, right)); }
MATH_FUNC_DEF quat& operator/=(quat& left, const float	right) { return(left = quat_div_f	(left, right)); }



MATH_FUNC_DEF bool operator==(const vec2	left, const vec2	right)	{ return(vec2_equals(left, right)); }
MATH_FUNC_DEF bool operator==(const vec3	left, const vec3	right)	{ return(vec3_equals(left, right)); }
MATH_FUNC_DEF bool operator==(const vec4	left, const vec4	right)	{ return(vec4_equals(left, right)); }

MATH_FUNC_DEF bool operator!=(const vec2	left, const vec2	right)	{ return(!vec2_equals(left, right)); }
MATH_FUNC_DEF bool operator!=(const vec3	left, const vec3	right)	{ return(!vec3_equals(left, right)); }
MATH_FUNC_DEF bool operator!=(const vec4	left, const vec4	right)	{ return(!vec4_equals(left, right)); }



MATH_FUNC_DEF vec2 operator-(const vec2 in)	{ return(vec2_negate(in)); }
MATH_FUNC_DEF vec3 operator-(const vec3 in)	{ return(vec3_negate(in)); }

MATH_FUNC_DEF vec4 operator-(const vec4 in) {
	vec4 result;
#if MATHLIB_USE_SIMD
	result._elements_sse = _mm_xor_ps(in._elements_sse, _mm_set1_ps(-0.0f));
#else
	result = (vec4){ -in.x, -in.y, -in.z, -in.w };
#endif
	return(result);
}

#endif // __cplusplus



#endif // MATHLIB_H_INCLUDED
