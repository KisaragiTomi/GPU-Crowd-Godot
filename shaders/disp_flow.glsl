#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { uint  cell_blocked_in[];      };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float goal_vx_in[];           };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { float goal_vy_in[];           };
layout(set = 0, binding = 3, std430) restrict writeonly buffer B3 { float disp_vx_out[];          };
layout(set = 0, binding = 4, std430) restrict writeonly buffer B4 { float disp_vy_out[];          };
layout(set = 0, binding = 5, std430) restrict readonly  buffer B5 { uint  cell_attacker_in[];     };
layout(set = 0, binding = 6, std430) restrict readonly  buffer B6 { uint  faction_presence_in[];  };
layout(set = 0, binding = 7, std430) restrict readonly  buffer B7 { uint  fac_to_group_in[];      };

layout(push_constant, std430) uniform Params {
	int field_gw;
	int field_gh;
	int num_groups;
	int num_factions;
};

void main() {
	int fx = int(gl_GlobalInvocationID.x);
	int fy = int(gl_GlobalInvocationID.y);
	int grp = int(gl_GlobalInvocationID.z);
	if (fx >= field_gw || fy >= field_gh || grp >= num_groups) return;

	int gi = fy * field_gw + fx;
	int gi_grp = grp * field_gw * field_gh + gi;

	if (cell_blocked_in[gi] == 0xFFFFFFFFu) {
		disp_vx_out[gi_grp] = 0.0;
		disp_vy_out[gi_grp] = 0.0;
		return;
	}

	uint enemy_fac_mask = 0u;
	for (int f = 0; f < num_factions; f++) {
		if (fac_to_group_in[f] != uint(grp)) {
			enemy_fac_mask |= (1u << uint(f));
		}
	}

	const int R = 3;
	float best_dsq = 1e10;
	float best_dx = 0.0, best_dy = 0.0;

	for (int dy = -R; dy <= R; dy++) {
		int ny = fy + dy;
		if (ny < 0 || ny >= field_gh) continue;
		for (int dx = -R; dx <= R; dx++) {
			if (dx == 0 && dy == 0) continue;
			int nx = fx + dx;
			if (nx < 0 || nx >= field_gw) continue;
			int ni = ny * field_gw + nx;

			if (cell_blocked_in[ni] == 0xFFFFFFFFu) continue;
			if (cell_attacker_in[ni] != 0xFFFFFFFFu) continue;
			if ((faction_presence_in[ni] & enemy_fac_mask) != 0u) continue;

			float dsq = float(dx * dx + dy * dy);
			if (dsq < best_dsq) {
				best_dsq = dsq;
				best_dx = float(dx);
				best_dy = float(dy);
			}
		}
	}

	if (best_dsq < 1e9) {
		float inv_d = inversesqrt(best_dsq);
		disp_vx_out[gi_grp] = best_dx * inv_d;
		disp_vy_out[gi_grp] = best_dy * inv_d;
	} else {
		int goal_idx = grp * field_gw * field_gh + gi;
		float sdx = -goal_vx_in[goal_idx];
		float sdy = -goal_vy_in[goal_idx];
		float slen = sqrt(sdx * sdx + sdy * sdy);
		if (slen > 0.01) {
			disp_vx_out[gi_grp] = sdx / slen;
			disp_vy_out[gi_grp] = sdy / slen;
		} else {
			disp_vx_out[gi_grp] = 0.0;
			disp_vy_out[gi_grp] = 0.0;
		}
	}
}
