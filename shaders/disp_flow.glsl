#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { uint  cell_blocked_in[];   };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float goal_vx_in[];       };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { float goal_vy_in[];       };
layout(set = 0, binding = 3, std430) restrict writeonly buffer B3 { float disp_vx_out[];      };
layout(set = 0, binding = 4, std430) restrict writeonly buffer B4 { float disp_vy_out[];      };
layout(set = 0, binding = 5, std430) restrict readonly  buffer B5 { uint  cell_attacker_in[];  };

layout(push_constant, std430) uniform Params {
	int field_gw;
	int field_gh;
	int num_groups;
	int _pad;
};

void main() {
	int fx = int(gl_GlobalInvocationID.x);
	int fy = int(gl_GlobalInvocationID.y);
	int grp = int(gl_GlobalInvocationID.z);
	if (fx >= field_gw || fy >= field_gh || grp >= num_groups) return;

	int gi = fy * field_gw + fx;
	int gi_grp = grp * field_gw * field_gh + gi;
	uint grp_bit = 1u << uint(grp);

	if (cell_blocked_in[gi] == 0xFFFFFFFFu) {
		disp_vx_out[gi_grp] = 0.0;
		disp_vy_out[gi_grp] = 0.0;
		return;
	}

	int goal_idx = grp * field_gw * field_gh + gi;
	float safe_dx = -goal_vx_in[goal_idx];
	float safe_dy = -goal_vy_in[goal_idx];
	float safe_len = sqrt(safe_dx * safe_dx + safe_dy * safe_dy);
	if (safe_len > 0.01) { safe_dx /= safe_len; safe_dy /= safe_len; }

	bool self_occupied = (cell_attacker_in[gi] != 0xFFFFFFFFu);

	const int R = 5;
	float away_enemy_x = 0.0, away_enemy_y = 0.0;
	float best_dsq = 1e10, best_dx = 0.0, best_dy = 0.0;
	float push_occ_x = 0.0, push_occ_y = 0.0;

	for (int dy = -R; dy <= R; dy++) {
		int ny = fy + dy;
		if (ny < 0 || ny >= field_gh) continue;
		for (int dx = -R; dx <= R; dx++) {
			if (dx == 0 && dy == 0) continue;
			int nx = fx + dx;
			if (nx < 0 || nx >= field_gw) continue;
			int ni = ny * field_gw + nx;

			if (cell_blocked_in[ni] == 0xFFFFFFFFu) continue;

			bool nb_enemy = (cell_blocked_in[ni] & grp_bit) != 0u;
			bool nb_empty = (cell_attacker_in[ni] == 0xFFFFFFFFu);

			float dsq = float(dx * dx + dy * dy);
			float w = 1.0 / max(dsq, 1.0);

			if (nb_enemy) {
				away_enemy_x -= float(dx) * w;
				away_enemy_y -= float(dy) * w;
			}

			if (self_occupied && nb_empty) {
				float dd = sqrt(dsq);
				float dir_dot = (safe_len > 0.01) ? (float(dx) * safe_dx + float(dy) * safe_dy) / max(dd, 0.01) : 1.0;
				if (dir_dot > 0.0 && dsq < best_dsq) {
					best_dsq = dsq;
					best_dx = float(dx);
					best_dy = float(dy);
				}
			}

			if (!self_occupied && !nb_empty) {
				push_occ_x -= float(dx) * w;
				push_occ_y -= float(dy) * w;
			}
		}
	}

	if (self_occupied) {
		float ox, oy;
		bool has_safe_empty = (best_dsq < 1e9);
		float enemy_len = sqrt(away_enemy_x * away_enemy_x + away_enemy_y * away_enemy_y);

		if (has_safe_empty) {
			float inv_d = inversesqrt(best_dsq);
			ox = best_dx * inv_d;
			oy = best_dy * inv_d;
			if (enemy_len > 0.01) {
				float enx = away_enemy_x / enemy_len;
				float eny = away_enemy_y / enemy_len;
				float b = clamp(enemy_len * 3.0, 0.0, 0.4);
				ox = ox * (1.0 - b) + enx * b;
				oy = oy * (1.0 - b) + eny * b;
			}
		} else if (enemy_len > 0.01) {
			ox = away_enemy_x / enemy_len;
			oy = away_enemy_y / enemy_len;
		} else if (safe_len > 0.01) {
			ox = safe_dx;
			oy = safe_dy;
		} else {
			ox = 0.0; oy = 0.0;
		}

		float olen = sqrt(ox * ox + oy * oy);
		if (olen > 0.01) {
			disp_vx_out[gi_grp] = ox / olen;
			disp_vy_out[gi_grp] = oy / olen;
		} else {
			disp_vx_out[gi_grp] = 0.0;
			disp_vy_out[gi_grp] = 0.0;
		}
	} else {
		float plen = sqrt(push_occ_x * push_occ_x + push_occ_y * push_occ_y);
		if (plen > 0.01) {
			disp_vx_out[gi_grp] = push_occ_x / plen;
			disp_vy_out[gi_grp] = push_occ_y / plen;
		} else {
			disp_vx_out[gi_grp] = 0.0;
			disp_vy_out[gi_grp] = 0.0;
		}
	}
}
