#include "stb_image.h"
#include "stb_image_write.h"
#include <stdio.h>

struct RT_Context {
	int resolution_x;
	int resolution_y;
};

int main() {
	RT_Context ctx = {};
	
	ctx.resolution_x = 720;
	ctx.resolution_y = 480;
	
	unsigned char* png_buf = (unsigned char*)malloc(ctx.resolution_x * ctx.resolution_y * 3);
	for(int x = 0; x < ctx.resolution_x; x++) {
		png_buf[x + 1000] = 0xFF;
	}
	
	
	stbi_write_png("test.png", ctx.resolution_x, ctx.resolution_y, 3, png_buf, 3);
	
	printf("hello world\n");
	return 0;
}