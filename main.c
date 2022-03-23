#include "stb_image.h"
#include "stb_image_write.h"
#include "mathlib.h"
#include <stdio.h>
#include <stdint.h>
#include <intrin.h>

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

size_t map_2d_index_to_1d(const int x, const int y, RT_Context* ctx) {
	return x + (ctx->resolution_y - y - 1) * ctx->resolution_x;
}

vec3 ray_pinhole_projection(vec2 uv, const RT_Context* ctx) {
	uv = vec2_mul_f(vec2_sub(uv, (vec2){0.5f, 0.5f}), 2.0f);
	const float fov_tan = f32_tan(ctx->fov_rad * 0.5f);
	return vec3_normalize((vec3){uv.x * fov_tan / ctx->aspect_y, uv.y * fov_tan, 1.0f});
}

rgb8 vec3_to_rgb8(const vec3 v) {
	return (rgb8){
		(uint8_t)(v.x * 255.0f),
		(uint8_t)(v.y * 255.0f),
		(uint8_t)(v.z * 255.0f),
	};
}

void renderScene(RT_Context* ctx) {
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
			
			ctx->out_image_data[index] = (rgb8){(uint8_t)(uv.u*255.0f), (uint8_t)(uv.v*255.0f), 0}; // draw UV
			ctx->out_image_data[index] = vec3_to_rgb8(vec3_mul_f(vec3_add(rd, vec3_init_f(1.0f)), 0.5f));
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
	
	renderScene(&ctx);
	
	stbi_write_png("image.png", ctx.resolution_x, ctx.resolution_y, 3, ctx.out_image_data, sizeof(rgb8)*ctx.resolution_x);
	
	printf("hello world\n");
	return 0;
}