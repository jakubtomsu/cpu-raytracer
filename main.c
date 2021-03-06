#include "stb_image.h"
#include "stb_image_write.h"
#include "sched.h"
#include "mathlib.h"
#include <stdio.h>
#include <stdint.h>
#include <intrin.h>
#include <time.h>
#include <assert.h>

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
	
	struct {
		vec3	dir;
		float	randomness;
		vec3	col;
		int	sample_num;
		vec3	add_col;
	} sun;
	
	struct {
		size_t total_scene_raycasts;
		size_t total_shape_intersection_tests;
	} debug;
} RT_Context;

#define RT_TASKS_ENABLED 1
#define RT_TASK_PIXELS_X 64
#define RT_TASK_PIXELS_Y 64
#define RT_TASK_NUM_X 32
#define RT_TASK_NUM_Y 18

#define RT_RESOLUTION_X (RT_TASK_PIXELS_X * RT_TASK_NUM_X)
#define RT_RESOLUTION_Y (RT_TASK_PIXELS_Y * RT_TASK_NUM_Y)


static inline size_t map_2d_index_to_1d(const int x, const int y, RT_Context* ctx) {
	return x + (ctx->resolution_y - y - 1) * ctx->resolution_x;
}

static inline vec3 ray_pinhole_projection(vec2 uv, const RT_Context* ctx) {
	uv = vec2_mul_f(vec2_sub(uv, (vec2){0.5f, 0.5f}), 2.0f);
	const float fov_tan = f32_tan(ctx->fov_rad*0.5f);
	return vec3_normalize((vec3){uv.x * fov_tan / ctx->aspect_y, uv.y * fov_tan, 1.0f});
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

static inline vec3 Ray_SphereNormal(const vec3 hit, const vec3 sph_center) {
	return vec3_normalize(vec3_sub(hit, sph_center));
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

// FIXME
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


typedef unsigned char RT_ShapeKind;
#define RT_SHAPEKIND_SPHERE 0
#define RT_SHAPEKIND_BOX 1

typedef struct {vec3 pos; float rad;} RT_Shape_Sphere;
typedef struct {vec3 pos; vec3 size;} RT_Shape_Box;

typedef struct RT_Shape {
	union {
		RT_Shape_Sphere sphere;
		RT_Shape_Box box;
	};
	vec3 col;
	RT_ShapeKind kind;
} RT_Shape;

static RT_Shape SceneShapes[1024] = {0};
static int SceneShapes_num = 0;

typedef struct Ray_Result {
	vec3 normal;
	float t;
	vec3 col;
	int is_valid_hit;
} Ray_Result;

static inline vec3 Ray_CalcSkyColor(const vec3 d) {
	const float dt = d.y*.5 + .5;
	vec3 result = (vec3){.1+dt*.5, .2+dt*.7, .1+dt*.9};
	result = vec3_mul_f(result, 0.5f);
	return result;
}

Ray_Result Ray_IntersectScene(const vec3 ro, const vec3 rd, RT_Context* ctx) {
	Ray_Result result = {0};
	result.t = 1e10f;
	
	for(int i = 0; i < SceneShapes_num; i++) {
		vec2 nf;
		RT_Shape shape = SceneShapes[i];
		vec3 normal = {0};
		switch(shape.kind) {
			case RT_SHAPEKIND_SPHERE: {
				nf = Ray_IntersectSphere(ro, rd, shape.sphere.pos, shape.sphere.rad);
				normal = Ray_SphereNormal(vec3_add(ro, vec3_mul_f(rd, nf.x)), shape.sphere.pos);
			} break;
			case RT_SHAPEKIND_BOX: {
				nf = Ray_IntersectBox(vec3_sub(ro, shape.box.pos), rd, shape.box.size, &normal);
			} break;
		}
	
		if(NearFar_Hit(nf) && nf.x > -0.0001f && nf.x < result.t) {
			result.is_valid_hit = 1;
			result.t = nf.x;
			result.normal = normal;
			result.col = shape.col;
		}
	}
	
	ctx->debug.total_scene_raycasts++;
	ctx->debug.total_shape_intersection_tests += SceneShapes_num;

	return result;
}



static inline vec3 dir_to_color(const vec3 dir) {
	return vec3_mul_f(vec3_add(dir, vec3_init_f(1.0f)), 0.5f);
}

static inline float randf32() {
	return (((float)rand()/(float)(RAND_MAX)) - 0.5f) * 2.0f;
}


static inline rgb8 vec3_to_rgb8(const vec3 v) {
	return (rgb8){
		(uint8_t)(f32_clamp(v.x, 1e-6f, 1.0f-1e-6f) * 255.0f),
		(uint8_t)(f32_clamp(v.y, 1e-6f, 1.0f-1e-6f) * 255.0f),
		(uint8_t)(f32_clamp(v.z, 1e-6f, 1.0f-1e-6f) * 255.0f),
	};
}

vec3 RandSphereDir() {
	return vec3_normalize((vec3){
		randf32(),
		randf32(),
		randf32(),
	});
}

vec3 RandHemisphereDir(const vec3 normal) {
	vec3 d = {0};
	if(1) {
		d = vec3_normalize(vec3_add(RandSphereDir(), vec3_mul_f(normal, 1.0f+1e-6f)));
	} else {
		d = RandSphereDir();
		if(vec3_dot(d, normal) < 0.0f) d = vec3_negate(d);
	}
	return d;
}

// @returns: color
vec3 RaytraceShadowRay(const vec3 p, const vec3 n, RT_Context* ctx) {
	const vec3 d = vec3_normalize(vec3_add(vec3_mul_f(RandSphereDir(), ctx->sun.randomness), ctx->sun.dir));
	Ray_Result rr = Ray_IntersectScene(p, d, ctx);
	vec3 radiance = vec3_init_f(0.0);
	if(!rr.is_valid_hit) {
		radiance = vec3_add(radiance, vec3_mul_f(ctx->sun.add_col, vec3_dot(n, d)));
	}
	return radiance;
}

// @returns: color
void RaytraceRecursive(vec3 ro, vec3 rd, const int max_bounces, RT_Context* ctx, vec3* out_col, vec3* out_radiance) {
	vec3 col = {0};
	vec3 radiance = {0};
	int accum = 0;
	for(int i = 0; i < max_bounces; i++) {
		const float strength = 1.0f / (float)(i+1);
		Ray_Result rr = Ray_IntersectScene(ro, rd, ctx);
		accum += strength;
		if(rr.is_valid_hit) {
			ro = vec3_add(vec3_mul_f(rd, rr.t), vec3_mul_f(rr.normal, 1e-6f));
			vec3 radi = vec3_mul_f(RaytraceShadowRay(ro, rr.normal, ctx), -vec3_dot(rd, rr.normal));
			
			rd = RandHemisphereDir(rr.normal);
			col = vec3_add(col, vec3_mul_f(rr.col, strength));
			radiance = vec3_add(radiance, vec3_mul_f(radi, strength));
		} else {
			col = vec3_add(col, vec3_mul_f(Ray_CalcSkyColor(rd), strength));
			goto loop_end;
		}
	}
	loop_end:
	
	col = vec3_div_f(col, accum);
	*out_col = col;
	*out_radiance = radiance;
}

void RenderScene(RT_Context* ctx, const int x0, const int x1, const int y0, const int y1) {
	const vec3 pos = (vec3){0.0, 0, -2};
	const vec3 ro = pos;

	for(int x = x0; x < x1; x++) {
		for(int y = y0; y < y1; y++) {
			vec2 uv = (vec2){
				(float)x / (float)ctx->resolution_x,
				(float)y / (float)ctx->resolution_y,
			};
			const size_t index = map_2d_index_to_1d(x, y, ctx);
			
			vec3 rd = ray_pinhole_projection(uv, ctx);
			//rd = (vec3){1.0, 1.0, 1.0};
			//printf("%f %f %f\n", rd.x, rd.y, rd.z);
			
			vec3 col = {0};
			Ray_Result prim = Ray_IntersectScene(ro, rd, ctx);
			if(prim.is_valid_hit) {
				const int gi_num = 10;
				const vec3 prim_hitpoint = vec3_add(vec3_add(ro, vec3_mul_f(rd, prim.t)), vec3_mul_f(prim.normal, 1e-7f));
				col = prim.col;

				vec3 radiance = {0};

				// GI
				for(int i = 0; i < gi_num; i++) {
					vec3 d = RandHemisphereDir(prim.normal);
					vec3 c = {0};
					vec3 r = {0};
					RaytraceRecursive(prim_hitpoint, d, 2, ctx, &c, &r);
					col      = vec3_add(col,      vec3_div_f(c, gi_num));
					radiance = vec3_add(radiance, vec3_div_f(r, gi_num));
				}
				col = vec3_mul_f(col, 0.5f);

				// sun light
				for(int i = 0; i < ctx->sun.sample_num; i++) {
					radiance = vec3_add(radiance, RaytraceShadowRay(prim_hitpoint, prim.normal, ctx));
				}
				
				col = vec3_mul(col, radiance);
				//col = radiance;
			} else {
				col = Ray_CalcSkyColor(rd);
			}
			
			
			// gamma correction
			col = (vec3){
				f32_pow(col.r, 0.4545),
				f32_pow(col.g, 0.4545),
				f32_pow(col.b, 0.4545),
			};
			
			//ctx->out_image_data[index] = (rgb8){(uint8_t)(uv.u*255.0f), (uint8_t)(uv.v*255.0f), 0}; // draw UV
			//ctx->out_image_data[index] = vec3_to_rgb8(dir_to_color(rd));
			ctx->out_image_data[index] = vec3_to_rgb8(col);
		}
	}
}

typedef struct RT_ParallelTaskParams {
	RT_Context* ctx;
	int x0;
	int x1;
	int y0;
	int y1;
} RT_ParallelTaskParams;

static void RT_sched_ParallelTask(void* pArg, struct scheduler* sched, struct sched_task_partition* p, sched_uint thread_num) {
	RT_ParallelTaskParams* params = (RT_ParallelTaskParams*)pArg;
	
	//printf("task thread %i\n", thread_num);
	//printf("x %i %i y %i %i\n", params->x0, params->x1, params->y0, params->y1);
	//return;
	
	RenderScene(params->ctx, params->x0, params->x1, params->y0, params->y1);

	//for(int x = params->x0; x < params->x1; x++) {
	//	for(int y = params->y0; y < params->y1; y++) {
	//		const size_t index = map_2d_index_to_1d(x, y, params->ctx);
	//		//params->ctx->out_image_data[index] = vec3_to_rgb8(vec3_hsv_to_rgb((vec3){((float)thread_num)*0.1f, 1, 1}));
	//		params->ctx->out_image_data[index] = vec3_to_rgb8((vec3){1, 0, 1});
	//	}
	//}
}


#define TIMED_BLOCK(name, ...) { \
    const clock_t _timed_block_begin_##name = clock(); \
    __VA_ARGS__ \
    printf(#name " time = %f ms\n", \
        (double)(clock()-_timed_block_begin_##name)*1000.0/CLOCKS_PER_SEC); \
}

int main() {
	RT_Context ctx = {0};
	
	ctx.fov = 80.0f;
	ctx.resolution_x = RT_RESOLUTION_X;
	ctx.resolution_y = RT_RESOLUTION_Y;
	ctx.aspect_y = (float)RT_RESOLUTION_Y / (float)RT_RESOLUTION_X;
	ctx.fov_rad = f32_to_rad(ctx.fov);
	ctx.out_image_data = (rgb8*)malloc(RT_RESOLUTION_X * RT_RESOLUTION_Y * sizeof(rgb8));
	
	ctx.sun.dir = vec3_normalize((vec3){-0.6, .4, -0.9});
	ctx.sun.randomness = 0.04f;
	ctx.sun.col = (vec3){.4,.4,.1};
	ctx.sun.sample_num = 4;
	ctx.sun.add_col = vec3_div_f(ctx.sun.col, ctx.sun.sample_num);

	// set rand() seed
	srand(time(NULL)<<1);

	// init scene
	{
		SceneShapes[0].kind = RT_SHAPEKIND_BOX;
		SceneShapes[0].box = (RT_Shape_Box){(vec3){ 0, 2, 1}, (vec3){3,1,2.1}};
		SceneShapes[0].col = (vec3){1,1,1};
		SceneShapes[1].kind = RT_SHAPEKIND_BOX;
		SceneShapes[1].box = (RT_Shape_Box){(vec3){ 0,-2, 0}, (vec3){3,1,3}};
		SceneShapes[1].col = (vec3){1,1,1};
		SceneShapes[2].kind = RT_SHAPEKIND_BOX;
		SceneShapes[2].box = (RT_Shape_Box){(vec3){ 0, 0, 2}, (vec3){1,1,1}};
		SceneShapes[2].col = (vec3){1,1,1};
		SceneShapes[3].kind = RT_SHAPEKIND_BOX;
		SceneShapes[3].box = (RT_Shape_Box){(vec3){-2, 0, 0}, (vec3){1,1,1}};
		SceneShapes[3].col = (vec3){0,1,0};
		SceneShapes[4].kind = RT_SHAPEKIND_BOX;
		SceneShapes[4].box = (RT_Shape_Box){(vec3){ 2, 0, 0}, (vec3){1,1,1}};
		SceneShapes[4].col = (vec3){1,0,0};
		
		SceneShapes[5].kind = RT_SHAPEKIND_SPHERE;
		SceneShapes[5].sphere = (RT_Shape_Sphere){(vec3){ 0, -0.6f, 0}, 0.3f};
		SceneShapes[5].col = (vec3){0,1,1};
		
		SceneShapes_num = 6;
	}
	
	
	printf("resolution = %ix%i  task_num=%i\n", RT_RESOLUTION_X, RT_RESOLUTION_Y, RT_TASK_NUM_X*RT_TASK_NUM_Y);

	TIMED_BLOCK(raytrace,
		if(0) {
			RenderScene(&ctx, 0, RT_RESOLUTION_X, 0, RT_RESOLUTION_Y);
		} else {
			void* memory;
			sched_size needed_memory;
			struct scheduler sched;
			scheduler_init(&sched, &needed_memory, SCHED_DEFAULT, 0);
			printf("scheduler memory = %i b\n", needed_memory);
			memory = calloc(needed_memory, 1);
			scheduler_start(&sched, memory);
			
			
			{
				struct sched_task sched_tasks[RT_TASK_NUM_X][RT_TASK_NUM_Y]; // !!!!!!!
				RT_ParallelTaskParams task_params[RT_TASK_NUM_X][RT_TASK_NUM_Y];

				for(int x = 0; x < RT_TASK_NUM_X; x++) {
					for(int y = 0; y < RT_TASK_NUM_Y; y++) {
						task_params[x][y].ctx = &ctx;
						task_params[x][y].x0 = x    *RT_TASK_PIXELS_X;
						task_params[x][y].x1 = (x+1)*RT_TASK_PIXELS_X;
						task_params[x][y].y0 = y    *RT_TASK_PIXELS_Y;
						task_params[x][y].y1 = (y+1)*RT_TASK_PIXELS_Y;
						
						//printf("scheduling task %i %i\n", x, y);
						
						scheduler_add(&sched, &sched_tasks[x][y], RT_sched_ParallelTask, &task_params[x][y], 1, 1);
					}
				}

				scheduler_wait(&sched);
			}
			
			scheduler_stop(&sched, 1);
			free(memory);
		}
	);
	
	printf("total scene raycasts = %lli\n", ctx.debug.total_scene_raycasts);
	printf("total shape intersection tests = %lli\n", ctx.debug.total_shape_intersection_tests);
	
	
	stbi_write_png("image.png", RT_RESOLUTION_X, RT_RESOLUTION_Y, 3, ctx.out_image_data, sizeof(rgb8)*RT_RESOLUTION_X);
	return 0;
}