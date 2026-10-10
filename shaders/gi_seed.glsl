#[compute]
#version 450

// 把 emission 累加缓冲（fp32 位模式）转成 RGBA16F emission 纹理，同时生成 JFA 种子
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer EmitAccBuffer {
	uvec4 emit_acc[];
};
layout(set = 0, binding = 1, rgba16f) restrict writeonly uniform image2D emit_image;
// 种子 = 像素坐标 x | y << 16，0 表示无种子（与原实现一样，(0,0) 处的种子也当作无）
layout(set = 0, binding = 2, r32ui) restrict writeonly uniform uimage2D seed_image;

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(emit_image);
	if (pos.x >= sz.x || pos.y >= sz.y) return;

	vec4 emit = uintBitsToFloat(emit_acc[pos.y * sz.x + pos.x]);
	imageStore(emit_image, pos, emit);
	// 阈值判断用 fp32 的 alpha，与原实现一致
	uint seed = emit.a > 0.01 ? (uint(pos.x) | (uint(pos.y) << 16)) : 0u;
	imageStore(seed_image, pos, uvec4(seed));
}
