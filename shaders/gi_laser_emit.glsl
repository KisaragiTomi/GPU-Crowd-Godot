#[compute]
#version 450

layout(local_size_x = 64) in;

layout(set = 0, binding = 0, std430) restrict readonly buffer B0 { int agent_info[]; };
layout(set = 0, binding = 1, std430) restrict readonly buffer B1 { vec4 fac_colors[32]; };
layout(set = 0, binding = 2, std430) restrict readonly buffer B2 { vec4 laser_line[]; };
layout(set = 0, binding = 3, std430) restrict readonly buffer B3 { float laser_ttl[]; };
layout(set = 0, binding = 4, rgba32f) uniform image2D emit_image;

layout(set = 0, binding = 5, std430) restrict readonly buffer ParamsBuffer {
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
};

vec2 world_to_tex(vec2 p) {
	return vec2(
		(p.x - region_x) / max(region_w, 0.001) * tex_w,
		(p.y - region_y) / max(region_h, 0.001) * tex_h
	);
}

void store_laser_pixel(ivec2 p, vec3 color, float alpha) {
	ivec2 sz = imageSize(emit_image);
	if (p.x < 0 || p.y < 0 || p.x >= sz.x || p.y >= sz.y) return;

	vec4 prev = imageLoad(emit_image, p);
	vec3 out_rgb = max(prev.rgb, color);
	float out_alpha = max(prev.a, alpha);
	imageStore(emit_image, p, vec4(out_rgb, out_alpha));
}

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (int(i) >= agent_count) return;

	float ttl = laser_ttl[i];
	if (ttl <= 0.0 || laser_life <= 0.0) return;

	uint info = uint(agent_info[i]);
	if ((info & 1u) == 0u) return;
	if ((info & (1u << 7u)) == 0u) return;

	vec4 line = laser_line[i];
	vec2 a = world_to_tex(line.xy);
	vec2 b = world_to_tex(line.zw);
	vec2 d = b - a;
	float len = length(d);
	if (len < 0.5) return;

	float life_frac = clamp(ttl / laser_life, 0.0, 1.0);
	uint fac = (info >> 1u) & 0x1Fu;
	vec3 fac_col = fac_colors[fac].rgb;
	vec3 laser_col = fac_col * laser_emit_strength * life_frac;

	float radius = max(laser_width, 0.05);
	float subpixel_coverage = clamp(radius, 0.02, 1.0);
	int ri = int(ceil(radius + 1.0));
	int steps = clamp(int(ceil(len * 1.25)), 1, 512);

	for (int s = 0; s <= steps; s++) {
		float t = float(s) / float(steps);
		vec2 center = mix(a, b, t);
		ivec2 ip = ivec2(floor(center + vec2(0.5)));
		for (int dy = -ri; dy <= ri; dy++) {
			for (int dx = -ri; dx <= ri; dx++) {
				vec2 delta = vec2(dx, dy) + vec2(ip) - center;
				float dist = length(delta);
				if (dist > radius + 0.75) continue;
				float falloff = smoothstep(radius + 0.75, 0.0, dist);
				float alpha = falloff * subpixel_coverage;
				store_laser_pixel(ip + ivec2(dx, dy), laser_col * alpha, alpha);
			}
		}
	}
}
