#[compute]
#version 450

// 远程攻击激光压缩：把可见激光追加到紧凑缓冲，CPU 一次回读后批量绘制
layout(local_size_x = 64) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { uint  agent_info[]; };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { vec4  fac_colors[32]; };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { vec4  laser_line[]; };
layout(set = 0, binding = 3, std430) restrict readonly  buffer B3 { float laser_ttl[]; };
// [alive, laser_count, 0, 0]，alive 由 build_multimesh 累加
layout(set = 0, binding = 4, std430) restrict           buffer B4 { uint  alive_counter; uint laser_count; };
// [点区 cap × (起点 xy, 终点 zw)][颜色区 cap × (阵营色 rgb, fade)]
layout(set = 0, binding = 5, std430) restrict writeonly buffer B5 { vec4  laser_out[]; };

layout(push_constant, std430) uniform Params {
	int   agent_count;
	uint  cap;
	float laser_life;
	float _pad0;
};

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (int(i) >= agent_count) return;

	float ttl = laser_ttl[i];
	if (ttl <= 0.0) return;
	vec4 line = laser_line[i];
	vec2 d = line.zw - line.xy;
	if (dot(d, d) < 1.0) return;  // 与原 CPU 绘制的跳过条件一致

	uint slot = atomicAdd(laser_count, 1u);
	if (slot >= cap) return;  // 超出容量只是不画，GI 仍读 per-agent 缓冲

	uint fac = (agent_info[i] >> 1u) & 0x1Fu;
	float fade = clamp(ttl / max(laser_life, 0.001), 0.0, 1.0);
	laser_out[slot] = line;
	laser_out[cap + slot] = vec4(fac_colors[fac].rgb, fade);
}
