#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { uint  faction_presence[]; };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float terrain[];          };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { uint  fac_to_group[];     };
layout(set = 0, binding = 3, std430) restrict           buffer B3 { uint  bfs_dist[];         };
layout(set = 0, binding = 4, std430) restrict writeonly buffer B4 { float goal_dist_all_out[];};
layout(set = 0, binding = 5, std430) restrict writeonly buffer B5 { float goal_dist_out[];    };
layout(set = 0, binding = 6, std430) restrict writeonly buffer B6 { float gdir_x_out[];       };
layout(set = 0, binding = 7, std430) restrict writeonly buffer B7 { float gdir_y_out[];       };

layout(push_constant, std430) uniform Params {
	int field_gw;
	int field_gh;
	int mode;
	int target_group;
	int num_factions;
	int _pad0;
	int _pad1;
	int _pad2;
};

void main() {
	int fx = int(gl_GlobalInvocationID.x);
	int fy = int(gl_GlobalInvocationID.y);
	if (fx >= field_gw || fy >= field_gh) return;
	int gi = fy * field_gw + fx;

	if (mode == 0) {
		// INIT: seed = enemy faction cells, wall = MAX
		if (terrain[gi] > 0.5) {
			bfs_dist[gi] = 0xFFFFFFFFu;
			return;
		}
		uint fp = faction_presence[gi];
		bool has_enemy = false;
		for (int f = 0; f < num_factions; f++) {
			if ((fp & (1u << uint(f))) != 0u && fac_to_group[f] != uint(target_group)) {
				has_enemy = true;
				break;
			}
		}
		bfs_dist[gi] = has_enemy ? 0u : 0xFFFFFFFEu;
	}
	else if (mode == 1) {
		// RELAX: 4-neighbor with atomicMin
		if (terrain[gi] > 0.5) return;
		uint cur = bfs_dist[gi];
		if (cur == 0u) return;
		uint best = cur;
		if (fx > 0)            { uint nd = bfs_dist[gi - 1];         if (nd < 0xFFFFFFFEu) best = min(best, nd + 1u); }
		if (fx < field_gw - 1) { uint nd = bfs_dist[gi + 1];         if (nd < 0xFFFFFFFEu) best = min(best, nd + 1u); }
		if (fy > 0)            { uint nd = bfs_dist[gi - field_gw];   if (nd < 0xFFFFFFFEu) best = min(best, nd + 1u); }
		if (fy < field_gh - 1) { uint nd = bfs_dist[gi + field_gw];   if (nd < 0xFFFFFFFEu) best = min(best, nd + 1u); }
		if (best < cur) atomicMin(bfs_dist[gi], best);
	}
	else if (mode == 2) {
		// GRADIENT: write float distance + gradient to output buffers
		int off = target_group * field_gw * field_gh;
		uint ud = bfs_dist[gi];
		float d = (ud >= 0xFFFFFFFEu) ? 1e6 : float(ud);
		goal_dist_all_out[off + gi] = d;
		if (target_group == 0) goal_dist_out[gi] = d;

		if (terrain[gi] > 0.5 || ud >= 0xFFFFFFFEu) {
			gdir_x_out[off + gi] = 0.0;
			gdir_y_out[off + gi] = 0.0;
			return;
		}

		float dl = (fx > 0)            ? float(min(bfs_dist[gi - 1],         0xFFFFFFFEu)) : d;
		float dr = (fx < field_gw - 1) ? float(min(bfs_dist[gi + 1],         0xFFFFFFFEu)) : d;
		float du = (fy > 0)            ? float(min(bfs_dist[gi - field_gw],   0xFFFFFFFEu)) : d;
		float dd = (fy < field_gh - 1) ? float(min(bfs_dist[gi + field_gw],   0xFFFFFFFEu)) : d;

		float ddx = dr - dl;
		float ddy = dd - du;
		float l = sqrt(ddx * ddx + ddy * ddy);
		if (l > 1e-6) {
			gdir_x_out[off + gi] = -ddx / l;
			gdir_y_out[off + gi] = -ddy / l;
		} else {
			gdir_x_out[off + gi] = 0.0;
			gdir_y_out[off + gi] = 0.0;
		}
	}
}
