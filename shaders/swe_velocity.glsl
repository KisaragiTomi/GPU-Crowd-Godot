#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { float fr[];      };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float fl[];      };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { float fd[];      };
layout(set = 0, binding = 3, std430) restrict readonly  buffer B3 { float fu[];      };
layout(set = 0, binding = 4, std430) restrict readonly  buffer B4 { float terrain[]; };
layout(set = 0, binding = 5, std430) restrict readonly  buffer B5 { float density[]; };
layout(set = 0, binding = 6, std430) restrict readonly  buffer B6 { float gdir_x[];  };
layout(set = 0, binding = 7, std430) restrict readonly  buffer B7 { float gdir_y[];  };
layout(set = 0, binding = 8, std430) restrict writeonly buffer B8 { float out_vx[];  };
layout(set = 0, binding = 9, std430) restrict writeonly buffer B9 { float out_vy[];  };
layout(set = 0, binding = 10, std430) restrict readonly buffer B10 { uint compact_tile_coords[]; };

layout(push_constant, std430) uniform Params {
	int   gw;
	int   gh;
	float agent_speed;
	float density_slowdown;
	int   sparse_mode;
	int   _pad0;
	int   _pad1;
	int   _pad2;
};

shared uint s_busy;

void main() {
	int lx = int(gl_LocalInvocationID.x);
	int ly = int(gl_LocalInvocationID.y);
	int li = int(gl_LocalInvocationIndex);

	int tile_x, tile_y;
	if (sparse_mode != 0) {
		uint packed = compact_tile_coords[gl_WorkGroupID.x];
		tile_x = int(packed & 0xFFFFu);
		tile_y = int(packed >> 16u);
	} else {
		tile_x = int(gl_WorkGroupID.x);
		tile_y = int(gl_WorkGroupID.y);
	}

	int x = tile_x * 8 + lx;
	int y = tile_y * 8 + ly;
	int group = int(gl_GlobalInvocationID.z);
	int gbase = group * gw * gh;

	if (li == 0) s_busy = 0u;
	barrier();

	if (x < gw && y < gh) {
		int i = y * gw + x;
		if (terrain[i] < 0.5) {
			float flux_sum = abs(fr[i]) + abs(fl[i]) + abs(fd[i]) + abs(fu[i]);
			if (density[i] > 0.001 || flux_sum > 0.01)
				atomicOr(s_busy, 1u);
		}
	}
	barrier();

	if (s_busy == 0u) {
		if (x < gw && y < gh) {
			int i = y * gw + x;
			if (terrain[i] > 0.5) {
				out_vx[gbase + i] = 0.0; out_vy[gbase + i] = 0.0;
			} else {
				out_vx[gbase + i] = gdir_x[gbase + i] * agent_speed;
				out_vy[gbase + i] = gdir_y[gbase + i] * agent_speed;
			}
		}
		return;
	}

	if (x >= gw || y >= gh) return;
	int i = y * gw + x;

	if (terrain[i] > 0.5) {
		out_vx[gbase + i] = 0.0; out_vy[gbase + i] = 0.0;
		return;
	}

	float fx   = fr[i] - fl[i];
	float fy   = fd[i] - fu[i];
	float fmag = sqrt(fx * fx + fy * fy);

	float gx = gdir_x[gbase + i];
	float gy = gdir_y[gbase + i];

	float dir_x = gx;
	float dir_y = gy;
	if (fmag > 0.01) {
		float avoidance = min(fmag * 0.05, 1.0);
		dir_x += (fx / fmag) * avoidance;
		dir_y += (fy / fmag) * avoidance;
	}
	float dlen = sqrt(dir_x * dir_x + dir_y * dir_y);
	if (dlen > 0.001) {
		dir_x /= dlen;
		dir_y /= dlen;
	}

	float local_speed = agent_speed / (1.0 + density[i] * density_slowdown);
	out_vx[gbase + i] = dir_x * local_speed;
	out_vy[gbase + i] = dir_y * local_speed;
}
