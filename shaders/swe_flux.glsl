#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { float density[];   };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float goal_dist[]; };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { float terrain[];   };
layout(set = 0, binding = 3, std430) restrict           buffer B3 { float fr[];        };
layout(set = 0, binding = 4, std430) restrict           buffer B4 { float fl[];        };
layout(set = 0, binding = 5, std430) restrict           buffer B5 { float fd[];        };
layout(set = 0, binding = 6, std430) restrict           buffer B6 { float fu[];        };
layout(set = 0, binding = 7, std430) restrict readonly  buffer B7 { uint compact_tile_coords[]; };

layout(push_constant, std430) uniform Params {
	int   gw;
	int   gh;
	float dt;
	float cell_size;
	float gravity;
	float damping;
	float density_scale;
	float goal_scale;
	float wall_scale;
	float max_flux;
	int   sparse_mode;
	float _pad1;
};

shared float s_H[10][10];
shared uint  s_active;

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

	int x  = tile_x * 8 + lx;
	int y  = tile_y * 8 + ly;
	int ox = tile_x * 8 - 1;
	int oy = tile_y * 8 - 1;

	float ds = density_scale;
	float gs = goal_scale;
	float ws = wall_scale;

	if (li == 0) s_active = 0u;
	barrier();

	for (int t = li; t < 100; t += 64) {
		int tx = t % 10;
		int ty = t / 10;
		int sx = clamp(ox + tx, 0, gw - 1);
		int sy = clamp(oy + ty, 0, gh - 1);
		int si = sy * gw + sx;
		float d = density[si];
		s_H[ty][tx] = d * ds + goal_dist[si] * gs + terrain[si] * ws;
		if (d > 0.001) atomicOr(s_active, 1u);
	}
	memoryBarrierShared();
	barrier();

	if (s_active == 0u) {
		if (x < gw && y < gh) {
			int i = y * gw + x;
			float dmp = damping;
			fr[i] *= dmp; fl[i] *= dmp; fd[i] *= dmp; fu[i] *= dmp;
		}
		return;
	}

	if (x >= gw || y >= gh) return;

	int   i    = y * gw + x;
	int   tlx  = lx + 1;
	int   tly  = ly + 1;
	float H    = s_H[tly][tlx];
	float coef = dt * gravity / cell_size;
	float dmp  = damping;
	float mf   = max_flux;

	if (x < gw - 1) {
		fr[i] = clamp(dmp * fr[i] + coef * (H - s_H[tly][tlx + 1]), 0.0, mf);
	} else { fr[i] = 0.0; }

	if (x > 0) {
		fl[i] = clamp(dmp * fl[i] + coef * (H - s_H[tly][tlx - 1]), 0.0, mf);
	} else { fl[i] = 0.0; }

	if (y < gh - 1) {
		fd[i] = clamp(dmp * fd[i] + coef * (H - s_H[tly + 1][tlx]), 0.0, mf);
	} else { fd[i] = 0.0; }

	if (y > 0) {
		fu[i] = clamp(dmp * fu[i] + coef * (H - s_H[tly - 1][tlx]), 0.0, mf);
	} else { fu[i] = 0.0; }
}
