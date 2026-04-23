#[compute]
#version 450
layout(local_size_x = 8, local_size_y = 8) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { float density[];          };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float terrain[];          };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { float goal_dist[];        };
layout(set = 0, binding = 3, std430) restrict readonly  buffer B3 { int   cell_blocked[];     };
layout(set = 0, binding = 4, std430) restrict writeonly buffer B4 { uint  rgba_out[];          };
layout(set = 0, binding = 5, std430) restrict readonly  buffer B5 { float out_vx[];           };
layout(set = 0, binding = 6, std430) restrict readonly  buffer B6 { float out_vy[];           };
layout(set = 0, binding = 7, std430) restrict readonly  buffer B7 { uint  faction_presence[]; };

layout(push_constant, std430) uniform Params {
	int   W;
	int   H;
	int   mode;
	int   param_i;
	float param_f;
	int   group_offset;
	int   faction_mask;
	int   _pad0;
};

vec3 hsv2rgb(float h, float s, float v) {
	float c = v * s;
	float x = c * (1.0 - abs(mod(h * 6.0, 2.0) - 1.0));
	float m = v - c;
	vec3 rgb;
	if      (h < 1.0/6.0) rgb = vec3(c, x, 0);
	else if (h < 2.0/6.0) rgb = vec3(x, c, 0);
	else if (h < 3.0/6.0) rgb = vec3(0, c, x);
	else if (h < 4.0/6.0) rgb = vec3(0, x, c);
	else if (h < 5.0/6.0) rgb = vec3(x, 0, c);
	else                   rgb = vec3(c, 0, x);
	return rgb + vec3(m);
}

void main() {
	uint x = gl_GlobalInvocationID.x;
	uint y = gl_GlobalInvocationID.y;
	if (int(x) >= W || int(y) >= H) return;
	uint idx = y * uint(W) + x;

	vec4 c = vec4(0.0);

	if (mode == 0) {
		float d = density[idx];
		if (d > 0.05 && (faction_mask == 0 || (int(faction_presence[idx]) & faction_mask) != 0))
			c = vec4(1.0, 0.15, 0.05, clamp(d * 0.25, 0.0, 0.7));
	} else if (mode == 1) {
		float gd = goal_dist[idx];
		if (terrain[idx] < 0.5 && gd < 1e5) {
			float t = 1.0 - gd / max(param_f, 1.0);
			c = vec4(0.05, t * 0.5, t * 0.35, 0.35);
		}
	} else if (mode == 2) {
		int cb = cell_blocked[idx];
		if (cb != -1 && (cb & param_i) != 0)
			c = vec4(1.0, 0.2, 0.2, 0.35);
	} else if (mode == 3) {
		const int SP = 6;
		int cx = (int(x) / SP) * SP + SP / 2;
		int cy = (int(y) / SP) * SP + SP / 2;
		if (cx >= 0 && cx < W && cy >= 0 && cy < H) {
			uint ci = uint(cy) * uint(W) + uint(cx);
			float vx = out_vx[group_offset + ci];
			float vy = out_vy[group_offset + ci];
			float mag = sqrt(vx * vx + vy * vy);
			if (mag > 2.0) {
				float nx = vx / mag;
				float ny = vy / mag;
				float dx = float(int(x) - cx);
				float dy = float(int(y) - cy);
				float u = dx * nx + dy * ny;
				float v = -dx * ny + dy * nx;
				float half_len = clamp(mag * 0.03, 1.0, float(SP / 2));
				if (abs(u) < half_len && abs(v) < 0.6)
					c = vec4(0.5, 0.8, 1.0, 0.75);
			}
		}
	}

	uint r = uint(clamp(c.r * 255.0, 0.0, 255.0));
	uint g = uint(clamp(c.g * 255.0, 0.0, 255.0));
	uint b = uint(clamp(c.b * 255.0, 0.0, 255.0));
	uint a = uint(clamp(c.a * 255.0, 0.0, 255.0));
	rgba_out[idx] = r | (g << 8) | (b << 16) | (a << 24);
}
