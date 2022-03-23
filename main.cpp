#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>
#include <stdint.h>

struct RT_Context {
	int resolution_x;
	int resolution_y;
};

size_t map_2d_index_to_1d(const int x, const int y, RT_Context* ctx) {
	return x + (ctx->resolution_y - y - 1) * ctx->resolution_x;
}

struct rgb8 {
	unsigned char r;
	unsigned char g;
	unsigned char b;
};

int main() {
	RT_Context ctx = {};
	
	ctx.resolution_x = 720;
	ctx.resolution_y = 480;
	
	rgb8* png_buf = (rgb8*)malloc(ctx.resolution_x * ctx.resolution_y * sizeof(rgb8));
	for(int x = 0; x < ctx.resolution_x; x++) {
		for (int y = 0; y < ctx.resolution_y; y++) {
			float u = float(x) / float(ctx.resolution_x) * 255.0f;
			float v = float(y) / float(ctx.resolution_y) * 255.0f;
			const size_t index = map_2d_index_to_1d(x, y, &ctx);
			png_buf[index] = { (unsigned char)u, (unsigned char)v, 0};
			//png_buf[index] = { 255, (unsigned char)(x > 100 ? 255 : 0), 0 };
		}
	}
	
	stbi_write_png("test.png", ctx.resolution_x, ctx.resolution_y, 3, png_buf, sizeof(rgb8)*ctx.resolution_x);
	
	printf("hello world\n");
	return 0;
}