#[compute]
#version 450

// 清空 emission 累加缓冲并写入墙体遮挡（原 gi_clear + gi_wall_mask 合并成一个逐像素 pass）
layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer TerrainBuffer {
	float terrain[];
};
// 每像素一个 uvec4：emission rgba 的 fp32 位模式（非负浮点按 uint 比较与按浮点比较同序，供 atomicMax）
layout(set = 0, binding = 1, std430) restrict writeonly buffer EmitAccBuffer {
	uvec4 emit_acc[];
};

layout(set = 0, binding = 2, std430) restrict readonly buffer ParamsBuffer {
	int   agent_count;
	float tex_w;
	float tex_h;
	float region_x;
	float region_y;
	float region_w;
	float region_h;
	float splat_radius;
	float emit_strength;
	int   terrain_w;
	int   terrain_h;
	float terrain_cell_size;
	float wall_alpha;
};

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = ivec2(int(tex_w), int(tex_h));
	if (pos.x >= sz.x || pos.y >= sz.y) return;

	float alpha = 0.0;
	if (terrain_w > 0 && terrain_h > 0 && terrain_cell_size > 0.0) {
		vec2 uv = (vec2(pos) + vec2(0.5)) / vec2(sz);
		vec2 world_pos = vec2(region_x, region_y) + uv * vec2(region_w, region_h);
		ivec2 cell = ivec2(floor(world_pos / terrain_cell_size));

		bool blocked = false;
		if (cell.x < 0 || cell.x >= terrain_w || cell.y < 0 || cell.y >= terrain_h) {
			blocked = true;
		} else {
			int idx = cell.y * terrain_w + cell.x;
			blocked = terrain[idx] > 0.5;
		}
		if (blocked) alpha = wall_alpha;
	}
	emit_acc[pos.y * sz.x + pos.x] = uvec4(0u, 0u, 0u, floatBitsToUint(alpha));
}
