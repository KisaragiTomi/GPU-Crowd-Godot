#[compute]
#version 450

layout(local_size_x = 64) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { float pos_x[]; };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float pos_y[]; };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { int agent_info[]; };
layout(set = 0, binding = 3, std430) restrict readonly  buffer B3 { int damage_acc[]; };
layout(set = 0, binding = 4, std430) restrict readonly  buffer B4 { int max_hp_arr[]; };
layout(set = 0, binding = 5, std430) restrict readonly  buffer B5 { vec4 fac_colors[32]; };
layout(set = 0, binding = 6, std430) restrict           buffer B6 { float disp_x[]; };
layout(set = 0, binding = 7, std430) restrict           buffer B7 { float disp_y[]; };
layout(set = 0, binding = 8, std430) restrict writeonly  buffer B8 { float mm_out[]; };
layout(set = 0, binding = 9, std430) restrict           buffer B9 { uint alive_counter; };

layout(push_constant, std430) uniform Params {
	int agent_count;
	float lerp_t;
	int _p0;
	int _p1;
};

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (int(i) >= agent_count) return;

	float px = pos_x[i];
	float py = pos_y[i];
	float dx = mix(disp_x[i], px, lerp_t);
	float dy = mix(disp_y[i], py, lerp_t);
	disp_x[i] = dx;
	disp_y[i] = dy;

	uint off = i * 12u;
	int ainfo = agent_info[i];

	if ((ainfo & 1) != 0) {
		int fac = (ainfo >> 1) & 0x1F;
		float inv_hp = 1.0 / max(float(max_hp_arr[i]), 1.0);
		float hp_frac = 1.0 - clamp(float(damage_acc[i]) * inv_hp, 0.0, 1.0);
		float brightness = mix(0.2, 1.0, hp_frac);
		vec4 col = fac_colors[fac];

		mm_out[off]      = 1.0; mm_out[off + 1u]  = 0.0;
		mm_out[off + 2u] = 0.0; mm_out[off + 3u]  = dx;
		mm_out[off + 4u] = 0.0; mm_out[off + 5u]  = 1.0;
		mm_out[off + 6u] = 0.0; mm_out[off + 7u]  = dy;
		mm_out[off + 8u]  = col.r * brightness;
		mm_out[off + 9u]  = col.g * brightness;
		mm_out[off + 10u] = col.b * brightness;
		mm_out[off + 11u] = 1.0;

		atomicAdd(alive_counter, 1u);
	} else {
		mm_out[off]      = 1.0; mm_out[off + 1u]  = 0.0;
		mm_out[off + 2u] = 0.0; mm_out[off + 3u]  = -10000.0;
		mm_out[off + 4u] = 0.0; mm_out[off + 5u]  = 1.0;
		mm_out[off + 6u] = 0.0; mm_out[off + 7u]  = -10000.0;
		mm_out[off + 8u]  = 0.0; mm_out[off + 9u]  = 0.0;
		mm_out[off + 10u] = 0.0; mm_out[off + 11u] = 0.0;
	}
}
