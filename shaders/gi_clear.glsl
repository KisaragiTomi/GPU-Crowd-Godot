#[compute]
#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba32f) writeonly uniform image2D output_image;

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(output_image);
	if (pos.x >= sz.x || pos.y >= sz.y) return;
	imageStore(output_image, pos, vec4(0.0, 0.0, 0.0, 0.0));
}
