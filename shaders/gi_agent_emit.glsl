#[compute]
#version 450

// GI emission（原 gi_agent_emit + gi_laser_emit 合并成一次 dispatch）：
// 线程 [0, agent_count) 各画一个 agent 光斑；之后每条激光 LASER_THREADS 个线程，各画该激光连续的一段采样点。
// 写入是对 fp32 位模式的 atomicMax（非负浮点按 uint 比较同序），重叠像素不再因"读 → 取 max → 写"丢更新。
layout(local_size_x = 64) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { float pos_x[]; };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float pos_y[]; };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { int agent_info[]; };
layout(set = 0, binding = 3, std430) restrict readonly  buffer B3 { vec4 fac_colors[32]; };
// 每像素 4 个 uint（rgba），由 gi_wall_mask 清空并写入墙体
layout(set = 0, binding = 4, std430) restrict buffer B4 { uint emit_acc[]; };
layout(set = 0, binding = 6, std430) restrict readonly  buffer B6 { int hit_flash[]; };
// GPUAgents.buf_laser_compact：[点区 laser_cap × (起点 xy, 终点 zw)][颜色区 laser_cap × (阵营色 rgb, fade)]
layout(set = 0, binding = 7, std430) restrict readonly  buffer B7 { vec4 laser_data[]; };

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
	int   laser_count;  // 本次要画的激光条数（已截到 laser_cap 以内）
	int   laser_cap;
};

const float HIT_FLASH_SCALE = 1024.0;
const float HIT_FLASH_COLOR_BOOST = 8.0;
const float HIT_FLASH_RADIUS_BOOST = 0.88;
const int LASER_THREADS = 16;  // 每条激光的线程数，须与 GPUGI.LASER_THREADS 一致

void emit_max(ivec2 p, vec3 rgb, float alpha) {
	uint base = uint(p.y * int(tex_w) + p.x) * 4u;
	if (rgb.r > 0.0) atomicMax(emit_acc[base], floatBitsToUint(rgb.r));
	if (rgb.g > 0.0) atomicMax(emit_acc[base + 1u], floatBitsToUint(rgb.g));
	if (rgb.b > 0.0) atomicMax(emit_acc[base + 2u], floatBitsToUint(rgb.b));
	if (alpha > 0.0) atomicMax(emit_acc[base + 3u], floatBitsToUint(alpha));
}

vec2 world_to_tex(vec2 p) {
	return vec2(
		(p.x - region_x) / max(region_w, 0.001) * tex_w,
		(p.y - region_y) / max(region_h, 0.001) * tex_h
	);
}

void splat_agent(uint i) {
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
	// 整个光斑都在纹理外：原实现逐像素越界跳过，结果相同
	if (cx + r < 0 || cy + r < 0 || cx - r >= iw || cy - r >= ih) return;
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
			emit_max(ivec2(sx, sy), col * alpha, alpha);
		}
	}
}

// 第 li 条激光的第 part 段。颜色 / 强度 / 宽度 / fade 规则与原 gi_laser_emit.glsl 相同（fade = clamp(ttl / laser_life, 0, 1)）
void stamp_laser(uint li, int part) {
	vec4 line = laser_data[li];
	vec4 col_fade = laser_data[uint(laser_cap) + li];
	vec2 a = world_to_tex(line.xy);
	vec2 b = world_to_tex(line.zw);
	vec2 d = b - a;
	float len = length(d);
	if (len < 0.5) return;

	vec3 laser_col = col_fade.rgb * laser_emit_strength * col_fade.a;

	float radius = max(laser_width, 0.05);
	float subpixel_coverage = clamp(radius, 0.02, 1.0);
	int ri = int(ceil(radius + 1.0));
	int steps = clamp(int(ceil(len * 1.25)), 1, 512);

	// 原实现一个线程依次画 s = 0..steps 共 steps + 1 个采样点，这里把它们均分给 LASER_THREADS 个线程
	int per = (steps + LASER_THREADS) / LASER_THREADS;
	int s0 = part * per;
	int s1 = min(s0 + per, steps + 1);
	if (s0 >= s1) return;
	// 这一段的包围盒完全在纹理外时整段跳过（原实现逐像素越界跳过，结果相同）
	vec2 c0 = mix(a, b, float(s0) / float(steps));
	vec2 c1 = mix(a, b, float(s1 - 1) / float(steps));
	vec2 lo = min(c0, c1) - float(ri + 1);
	vec2 hi = max(c0, c1) + float(ri + 1);
	if (hi.x < 0.0 || hi.y < 0.0 || lo.x >= tex_w || lo.y >= tex_h) return;

	int iw = int(tex_w);
	int ih = int(tex_h);
	for (int s = s0; s < s1; s++) {
		float t = float(s) / float(steps);
		vec2 center = mix(a, b, t);
		ivec2 ip = ivec2(floor(center + vec2(0.5)));
		for (int dy = -ri; dy <= ri; dy++) {
			for (int dx = -ri; dx <= ri; dx++) {
				ivec2 p = ip + ivec2(dx, dy);
				if (p.x < 0 || p.y < 0 || p.x >= iw || p.y >= ih) continue;
				vec2 delta = vec2(dx, dy) + vec2(ip) - center;
				float dist = length(delta);
				if (dist > radius + 0.75) continue;
				float falloff = smoothstep(radius + 0.75, 0.0, dist);
				float alpha = falloff * subpixel_coverage;
				emit_max(p, laser_col * alpha, alpha);
			}
		}
	}
}

void main() {
	uint t = gl_GlobalInvocationID.x;
	if (int(t) < agent_count) {
		splat_agent(t);
		return;
	}
	// 超过 laser_cap（GPUAgents.LASER_CAP = 8192）条的激光不在紧凑列表里，不发光
	uint k = t - uint(max(agent_count, 0));
	uint li = k / uint(LASER_THREADS);
	if (int(li) >= laser_count) return;
	stamp_laser(li, int(k % uint(LASER_THREADS)));
}
