#include "stb_image.h"
#include "stb_image_write.h"
#include "mathlib.h"
#include <stdio.h>
#include <stdint.h>
#include <intrin.h>

// some shape intersectors are from: https://www.iquilezles.org/www/articles/intersectors/intersectors.htm



#if !defined(PI)
	#define PI 3.14159265359f
#endif

typedef struct rgb8 {
	uint8_t r;
	uint8_t g;
	uint8_t b;
} rgb8;

typedef struct RT_Context {
	int resolution_x;
	int resolution_y;
	float aspect_y;
	float fov; // deg
	float fov_rad;
	rgb8* out_image_data;
} RT_Context;



static inline size_t map_2d_index_to_1d(const int x, const int y, RT_Context* ctx) {
	return x + (ctx->resolution_y - y - 1) * ctx->resolution_x;
}

static inline vec3 ray_pinhole_projection(vec2 uv, const RT_Context* ctx) {
	uv = vec2_mul_f(vec2_sub(uv, (vec2){0.5f, 0.5f}), 2.0f);
	const float fov_tan = f32_tan(ctx->fov_rad * 0.5f);
	return vec3_normalize((vec3){uv.x * fov_tan / ctx->aspect_y, uv.y * fov_tan, 1.0f});
}

static inline rgb8 vec3_to_rgb8(const vec3 v) {
	return (rgb8){
		(uint8_t)(v.x * 255.0f),
		(uint8_t)(v.y * 255.0f),
		(uint8_t)(v.z * 255.0f),
	};
}

// also returns true when inside
static inline bool NearFar_Hit(const vec2 t) {
	return t.x < t.y && t.y > 0;
}

static inline vec2 Ray_IntersectSphere(const vec3 ro, const vec3 rd, const vec3 sph_center, const float sph_rad) {
	const vec3 oc = vec3_sub(ro, sph_center);
	const float b = vec3_dot(oc, rd);
	const float c = vec3_dot(oc, oc) - sph_rad*sph_rad;
	float h = b*b - c;
	if(h < 0.0f) return (vec2){-1.0f, -1.0f}; // no intersection
	h = f32_sqrt(h);
	return (vec2){-b-h, -b+h};
}

static inline vec3 vec3_sign(const vec3 v) {
	return (vec3){
		v.x >= 0.0f ? 1.0f : -1.0f,
		v.y >= 0.0f ? 1.0f : -1.0f,
		v.z >= 0.0f ? 1.0f : -1.0f,
	};
}

// from GLSL
// https://www.khronos.org/registry/OpenGL-Refpages/gl4/html/step.xhtml
static inline vec3 vec3_step(const vec3 edge, const vec3 x) {
	return (vec3){
		x.x < edge.x ? 0.0 : 1.0,
		x.y < edge.y ? 0.0 : 1.0,
		x.z < edge.z ? 0.0 : 1.0,
	};
}

static inline vec3 vec3_div_safe(const vec3 a, const vec3 b) {
	return (vec3){
		b.x == 0.0f ? 1e10f : a.x / b.x,
		b.y == 0.0f ? 1e10f : a.y / b.y,
		b.z == 0.0f ? 1e10f : a.z / b.z,
	};
}

// axis aligned box centered at the origin, with size boxSize
static inline vec2 Ray_IntersectBox(const vec3 ro, const vec3 rd, const vec3 boxSize, vec3* out_normal) {
	const vec3 m = vec3_div_safe((vec3){1,1,1}, rd); // can precompute if traversing a set of aligned boxes
	const vec3 n = vec3_mul(m, ro);   // can precompute if traversing a set of aligned boxes
	const vec3 k = vec3_mul(vec3_abs(m), boxSize);
	const vec3 t1 = vec3_sub(vec3_negate(n), k);
	const vec3 t2 = vec3_add(vec3_negate(n), k);
	const float tN = f32_max(f32_max(t1.x, t1.y), t1.z);
	const float tF = f32_min(f32_min(t2.x, t2.y), t2.z);
	if(tN > tF || tF < 0.0) return (vec2){-1.0f, -1.0f};
	const vec3 t1_yzx = (vec3){t1.y, t1.z, t1.x};
	const vec3 t1_zxy = (vec3){t1.z, t1.x, t1.y};
	*out_normal = vec3_mul(vec3_mul(vec3_negate(vec3_sign(rd)), vec3_step(t1_yzx, t1)), vec3_step(t1_zxy, t1));
	return (vec2){tN, tF};
}

static inline float Ray_GoursatIntersect(const vec3 ro, const vec3 rd, float ka, float kb) {
	float po = 1.0;
	vec3 rd2 = vec3_mul(rd, rd); vec3 rd3 = vec3_mul(rd2, rd);
	vec3 ro2 = vec3_mul(ro, ro); vec3 ro3 = vec3_mul(ro2, ro);
	float k4 = vec3_dot(rd2,rd2);
	float k3 = vec3_dot(ro ,rd3);
	float k2 = vec3_dot(ro2,rd2) - kb/6.0;
	float k1 = vec3_dot(ro3,rd ) - kb*vec3_dot(rd,ro)/2.0;
	float k0 = vec3_dot(ro2,ro2) + ka - kb*vec3_dot(ro,ro);
	k3 /= k4;
	k2 /= k4;
	k1 /= k4;
	k0 /= k4;
	float c2 = k2 - k3*(k3);
	float c1 = k1 + k3*(2.0*k3*k3-3.0*k2);
	float c0 = k0 + k3*(k3*(c2+k2)*3.0-4.0*k1);

	if(f32_abs(c1) < 0.1f*f32_abs(c2)) {
		po = -1.0;
		float tmp=k1; k1=k3; k3=tmp;
		k0 = 1.0/k0;
		k1 = k1*k0;
		k2 = k2*k0;
		k3 = k3*k0;
		c2 = k2 - k3*(k3);
		c1 = k1 + k3*(2.0*k3*k3-3.0*k2);
		c0 = k0 + k3*(k3*(c2+k2)*3.0-4.0*k1);
	}

	c0 /= 3.0f;
	float Q = c2*c2 + c0;
	float R = c2*c2*c2 - 3.0f*c0*c2 + c1*c1;
	float h = R*R - Q*Q*Q;

	if(h>0.0f) { // 2 intersections
		h = f32_sqrt(h);
		float s = f32_sign(R+h)*f32_pow(f32_abs(R+h),1.0f/3.0f); // cube root
		float u = f32_sign(R-h)*f32_pow(f32_abs(R-h),1.0f/3.0f); // cube root
		float x = s+u+4.0f*c2;
		float y = s-u;
		float ks = x*x + y*y*3.0;
		float k = f32_sqrt(ks);
		float t = -0.5f*po*f32_abs(y)*f32_sqrt(6.0f/(k+x)) - 2.0f*c1*(k+x)/(ks+x*k) - k3;
		return (po<0.0f)?1.0f/t:t;
	}

	// 4 intersections
	float sQ = f32_sqrt(Q);
	float w = sQ*f32_cos(f32_acos(-R/(sQ*Q))/3.0f);
	float d2 = -w - c2; 
	if(d2<0.0) return -1.0; //no intersection
	float d1 = f32_sqrt(d2);
	float h1 = f32_sqrt(w - 2.0*c2 + c1/d1);
	float h2 = f32_sqrt(w - 2.0*c2 - c1/d1);
	float t1 = -d1 - h1 - k3; t1 = (po<0.0)?1.0/t1:t1;
	float t2 = -d1 + h1 - k3; t2 = (po<0.0)?1.0/t2:t2;
	float t3 =  d1 - h2 - k3; t3 = (po<0.0)?1.0/t3:t3;
	float t4 =  d1 + h2 - k3; t4 = (po<0.0)?1.0/t4:t4;
	float t = 1e20;
	if(t1>0.0f) t=t1;
	if(t2>0.0f) t=f32_min(t,t2);
	if(t3>0.0f) t=f32_min(t,t3);
	if(t4>0.0f) t=f32_min(t,t4);
	return t;
}

static inline vec3 Ray_GousatNormal(const vec3 pos, const float ka, const float kb) {
	return vec3_normalize(vec3_sub(vec3_mul_f(vec3_mul(vec3_mul(pos,pos),pos), 4.0f), vec3_mul_f(pos, kb*2.0f)));
}



typedef struct Ray_Result {
	vec3 normal;
	float t;
	vec3 col;
	int is_valid_hit;
} Ray_Result;

Ray_Result Ray_IntersectScene(const vec3 ro, const vec3 rd, RT_Context* ctx) {
	Ray_Result result = {0};

	vec2 nf = {0};
	float t = 0;
	// const vec2 nf = Ray_IntersectSphere(ro, rd, (vec3){0, 0, 10}, 4.0f);
	//nf = Ray_IntersectBox(vec3_sub(ro, (vec3){-1, -1.5, 3}), rd, (vec3){3, 1, 2}, &result.normal);
	const vec3 center = (vec3){-1, 1, -4000};
	t = Ray_GoursatIntersect(vec3_sub(ro, center), rd, 1.0f, 1.0f);
	const vec3 hitpos = vec3_add(ro, vec3_mul_f(rd, t));
	result.normal = Ray_GousatNormal(hitpos, 1.0f, 1.0f);
	

	// if(NearFar_Hit(nf)) { t = nf.x
	if(t > 0) {
		result.is_valid_hit = 1;
		result.t = t;
		result.col = vec3_init_f(t * 0.02f);
	}

	return result;
}



static inline vec3 dir_to_color(const vec3 dir) {
	return vec3_mul_f(vec3_add(dir, vec3_init_f(1.0f)), 0.5f);
}

void RenderScene(RT_Context* ctx) {
	for(int x = 0; x < ctx->resolution_x; x++) {
		for(int y = 0; y < ctx->resolution_y; y++) {
			vec2 uv = (vec2){
				(float)x / (float)ctx->resolution_x,
				(float)y / (float)ctx->resolution_y,
			};
			const size_t index = map_2d_index_to_1d(x, y, ctx);
			
			vec3 rd = ray_pinhole_projection(uv, ctx);
			//rd = (vec3){1.0, 1.0, 1.0};
			//printf("%f %f %f\n", rd.x, rd.y, rd.z);
			
			vec3 col = {0};
			Ray_Result rr = Ray_IntersectScene((vec3){0}, rd, ctx);
			col = rr.col;
			col = dir_to_color(rr.normal);
			
			// gamma correction
			col = (vec3){
				f32_pow(col.r, 0.4545),
				f32_pow(col.g, 0.4545),
				f32_pow(col.b, 0.4545),
			};
			
			ctx->out_image_data[index] = (rgb8){(uint8_t)(uv.u*255.0f), (uint8_t)(uv.v*255.0f), 0}; // draw UV
			ctx->out_image_data[index] = vec3_to_rgb8(dir_to_color(rd));
			ctx->out_image_data[index] = vec3_to_rgb8(col);
		}
	}
}

int main() {
	RT_Context ctx = {0};
	
	ctx.resolution_x = 720;
	ctx.resolution_y = 480;
	ctx.fov = 180.0f;
	
	ctx.aspect_y = (float)ctx.resolution_y / (float)ctx.resolution_x;
	ctx.fov_rad = f32_to_deg(ctx.fov);
	ctx.out_image_data = (rgb8*)malloc(ctx.resolution_x * ctx.resolution_y * sizeof(rgb8));
	
	RenderScene(&ctx);
	
	stbi_write_png("image.png", ctx.resolution_x, ctx.resolution_y, 3, ctx.out_image_data, sizeof(rgb8)*ctx.resolution_x);
	
	printf("hello world\n");
	return 0;
}