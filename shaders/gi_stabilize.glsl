#[compute]
#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba16f) readonly uniform image2D current_image;
layout(set = 0, binding = 1, rgba16f) readonly uniform image2D history_image;
// history 保持浮点：8 bit 会让指数衰减卡在最低档留下残影
layout(set = 0, binding = 2, rgba16f) writeonly uniform image2D output_image;
// 同一结果另存一份 RGBA8 供回读显示（GI 已 tonemap 到 [0, 1)）
layout(set = 0, binding = 4, rgba8) writeonly uniform image2D display_image;

layout(set = 0, binding = 3, std430) restrict readonly buffer GIParamsBuffer {
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
	float laser_width;
	float laser_emit_strength;
	float laser_life;
	int   laser_count;
	int   laser_cap;
	float interval_scale;
	float history_weight;
	float spatial_strength;
};

vec3 read_current(ivec2 pos, ivec2 sz) {
	return imageLoad(current_image, clamp(pos, ivec2(0), sz - ivec2(1))).rgb;
}

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(output_image);
	if (pos.x >= sz.x || pos.y >= sz.y) return;

	vec3 center = read_current(pos, sz);
	vec3 accum = center * 4.0;
	accum += read_current(pos + ivec2(-1, 0), sz) * 2.0;
	accum += read_current(pos + ivec2(1, 0), sz) * 2.0;
	accum += read_current(pos + ivec2(0, -1), sz) * 2.0;
	accum += read_current(pos + ivec2(0, 1), sz) * 2.0;
	accum += read_current(pos + ivec2(-1, -1), sz);
	accum += read_current(pos + ivec2(1, -1), sz);
	accum += read_current(pos + ivec2(-1, 1), sz);
	accum += read_current(pos + ivec2(1, 1), sz);

	vec3 blurred = accum / 16.0;
	vec3 filtered = mix(center, blurred, clamp(spatial_strength, 0.0, 1.0));
	vec3 history = imageLoad(history_image, pos).rgb;
	vec3 stable = mix(filtered, history, clamp(history_weight, 0.0, 0.98));

	imageStore(output_image, pos, vec4(stable, 1.0));
	imageStore(display_image, pos, vec4(stable, 1.0));
}
