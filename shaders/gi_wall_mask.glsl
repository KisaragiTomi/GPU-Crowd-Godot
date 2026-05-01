#[compute]
#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer TerrainBuffer {
	float terrain[];
};
layout(set = 0, binding = 1, rgba32f) uniform image2D emit_image;

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
	ivec2 sz = imageSize(emit_image);
	if (pos.x >= sz.x || pos.y >= sz.y) return;
	if (terrain_w <= 0 || terrain_h <= 0 || terrain_cell_size <= 0.0) return;

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

	if (blocked) {
		imageStore(emit_image, pos, vec4(0.0, 0.0, 0.0, wall_alpha));
	}
}
