#[compute]
#version 450

layout(local_size_x = 64) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { float pos_x[]; };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float pos_y[]; };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { int agent_info[]; };
layout(set = 0, binding = 3, std430) restrict readonly  buffer B3 { vec4 fac_colors[32]; };
layout(set = 0, binding = 4, rgba32f) uniform image2D emit_image;
layout(set = 0, binding = 6, std430) restrict readonly  buffer B6 { int hit_flash[]; };

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
};

const float HIT_FLASH_SCALE = 1024.0;
const float HIT_FLASH_COLOR_BOOST = 8.0;
const float HIT_FLASH_RADIUS_BOOST = 0.88;

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (int(i) >= agent_count) return;

	int ainfo = agent_info[i];
	if ((ainfo & 1) == 0) return;         // dead

	int fac = (ainfo >> 1) & 0x1F;
	float flash = clamp(float(hit_flash[i]) / HIT_FLASH_SCALE, 0.0, 1.0);
	vec3 col = fac_colors[fac].rgb * emit_strength * (1.0 + flash * HIT_FLASH_COLOR_BOOST);

	float px = pos_x[i];
	float py = pos_y[i];

	vec2 tpos = vec2(
		(px - region_x) / max(region_w, 0.001) * tex_w,
		(py - region_y) / max(region_h, 0.001) * tex_h
	);

	float rr = max(0.05, splat_radius * (1.0 + flash * HIT_FLASH_RADIUS_BOOST));
	int r = max(1, int(ceil(rr + 1.0)));
	int cx = int(floor(tpos.x));
	int cy = int(floor(tpos.y));
	int iw = int(tex_w);
	int ih = int(tex_h);
	float edge = rr + 0.75;
	float subpixel_coverage = clamp(rr * rr, 0.02, 1.0);

	for (int dy = -r; dy <= r; dy++) {
		for (int dx = -r; dx <= r; dx++) {
			int sx = cx + dx;
			int sy = cy + dy;
			if (sx < 0 || sx >= iw || sy < 0 || sy >= ih) continue;
			float dist = length(vec2(float(sx) + 0.5, float(sy) + 0.5) - tpos);
			if (dist > edge) continue;
			float alpha = smoothstep(edge, 0.0, dist) * subpixel_coverage;
			vec4 prev = imageLoad(emit_image, ivec2(sx, sy));
			imageStore(emit_image, ivec2(sx, sy), vec4(max(prev.rgb, col * alpha), max(prev.a, alpha)));
		}
	}
}
