#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <stdint.h>
#include <intrin.h>

typedef uint8_t u8;
typedef uint32_t i32;
typedef uint64_t isize;

struct RT_Context {
	i32 resolution_x;
	i32 resolution_y;
};

size_t map_2d_index_to_1d(const i32 x, const i32 y, RT_Context* ctx) {
	return x + (ctx->resolution_y - y - 1) * ctx->resolution_x;
}

struct rgb8 {
	u8 r;
	u8 g;
	u8 b;
};

struct RT_PixelLane {
	__m128 R;
	__m128 G;
	__m128 B;
};

i32 main() {
	RT_Context ctx = {};
	
	ctx.resolution_x = 720;
	ctx.resolution_y = 480;
	
	rgb8* png_buf = (rgb8*)malloc(ctx.resolution_x * ctx.resolution_y * sizeof(rgb8));
	for(i32 x = 0; x < ctx.resolution_x; x++) {
		for (i32 y = 0; y < ctx.resolution_y; y++) {
			float u = float(x) / float(ctx.resolution_x) * 255.0f;
			float v = float(y) / float(ctx.resolution_y) * 255.0f;
			const size_t index = map_2d_index_to_1d(x, y, &ctx);
			png_buf[index] = { (u8)u, (u8)v, 0};
		}
	}
	
	stbi_write_png("test.png", ctx.resolution_x, ctx.resolution_y, 3, png_buf, sizeof(rgb8)*ctx.resolution_x);
	
	printf("hello world\n");
	return 0;
}