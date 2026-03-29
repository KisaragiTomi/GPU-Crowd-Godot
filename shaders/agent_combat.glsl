#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0,  std430) restrict readonly  buffer B0  { float pos_x[];            };
layout(set = 0, binding = 1,  std430) restrict readonly  buffer B1  { float pos_y[];            };
layout(set = 0, binding = 2,  std430) restrict readonly  buffer B2  { uint  cell_start_buf[];   };
layout(set = 0, binding = 3,  std430) restrict readonly  buffer B3  { uint  cell_count_buf[];   };
layout(set = 0, binding = 4,  std430) restrict readonly  buffer B4  { uint  sorted_idx[];       };
layout(set = 0, binding = 5,  std430) restrict           buffer B5  { uint  agent_info[];       };
layout(set = 0, binding = 6,  std430) restrict           buffer B6  { int   damage_acc[];       };
layout(set = 0, binding = 7,  std430) restrict           buffer B7  { float cooldown[];         };
layout(set = 0, binding = 8,  std430) restrict readonly  buffer B8  { float atk_range[];        };
layout(set = 0, binding = 9,  std430) restrict readonly  buffer B9  { float atk_damage[];       };
layout(set = 0, binding = 10, std430) restrict readonly  buffer B10 { int   max_hp[];           };
layout(set = 0, binding = 11, std430) restrict readonly  buffer B11 { float regen_rate[];       };
layout(set = 0, binding = 12, std430) restrict readonly  buffer B12 { uint  faction_presence[];  };
layout(set = 0, binding = 13, std430) restrict readonly  buffer B13 { float goal_dist_all[];    };
layout(set = 0, binding = 14, std430) restrict readonly  buffer B14 { uint  alliance[];         };
layout(set = 0, binding = 15, std430) restrict readonly  buffer B15 { uint  fac_to_group[];     };

layout(push_constant, std430) uniform Params {
	int   agent_count;   // 0
	int   sg_w;          // 4
	int   sg_h;          // 8
	float sg_inv_cs;     // 12
	float dt;            // 16
	float engage_range;  // 20
	float attack_cd_base;// 24
	float inv_field_cs;  // 28
	int   field_gw;      // 32
	int   field_gh;      // 36
	int   field_cells;   // 40  (gw*gh, for goal_dist offset)
	uint  _pad0;         // 44
};

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (int(i) >= agent_count) return;

	uint info = agent_info[i];
	if ((info & 1u) == 0u) return;

	uint faction   = (info >> 1u) & 0x1Fu;
	uint ally_bits = alliance[faction];
	uint enemy_msk = ~ally_bits;

	// --- regen ---
	int regen_int = int(regen_rate[i] * dt * 100.0);
	if (regen_int > 0) {
		atomicAdd(damage_acc[i], -regen_int);
		atomicMax(damage_acc[i], 0);
	}

	// --- distance field gate (with real-time fallback) ---
	float px = pos_x[i];
	float py = pos_y[i];
	float ifc = inv_field_cs;
	int gi_x = clamp(int(px * ifc), 0, field_gw - 1);
	int gi_y = clamp(int(py * ifc), 0, field_gh - 1);
	int gi   = gi_y * field_gw + gi_x;

	uint group = fac_to_group[faction];
	float gdist = goal_dist_all[int(group) * field_cells + gi];
	float range_cells = atk_range[i] * ifc + 2.0;
	if (gdist > range_cells) {
		// Distance field says far — check faction_presence as real-time fallback
		bool enemies_adjacent = false;
		for (int fy = max(gi_y - 1, 0); fy <= min(gi_y + 1, field_gh - 1) && !enemies_adjacent; fy++) {
			for (int fx = max(gi_x - 1, 0); fx <= min(gi_x + 1, field_gw - 1); fx++) {
				if ((faction_presence[fy * field_gw + fx] & enemy_msk) != 0u) {
					enemies_adjacent = true;
					break;
				}
			}
		}
		if (!enemies_adjacent) {
			cooldown[i] = max(cooldown[i] - dt, 0.0);
			if (cooldown[i] <= 0.0)
				agent_info[i] = info & ~(1u << 9u);
			return;
		}
	}

	// --- neighbor scan (radius scales with attack range) ---
	float my_range = atk_range[i];
	float my_range_sq = my_range * my_range;
	int scan_r = clamp(int(ceil(my_range * sg_inv_cs)), 1, 4);

	uint  best_enemy = 0xFFFFFFFFu;
	float best_dsq   = 1e10;

	int hcx = clamp(int(px * sg_inv_cs), 0, sg_w - 1);
	int hcy = clamp(int(py * sg_inv_cs), 0, sg_h - 1);
	for (int ny = max(hcy - scan_r, 0); ny <= min(hcy + scan_r, sg_h - 1); ny++) {
		int row = ny * sg_w;
		for (int nx = max(hcx - scan_r, 0); nx <= min(hcx + scan_r, sg_w - 1); nx++) {
			int ci  = row + nx;
			int s   = int(cell_start_buf[ci]);
			int cnt = int(cell_count_buf[ci]);
			for (int k = 0; k < cnt; k++) {
				uint j = sorted_idx[s + k];
				if (j == i) continue;
				uint ji = agent_info[j];
				if ((ji & 1u) == 0u) continue;
				uint jf = (ji >> 1u) & 0x1Fu;
				if (((1u << jf) & enemy_msk) == 0u) continue;
				float djx = px - pos_x[j];
				float djy = py - pos_y[j];
				float dsq = djx * djx + djy * djy;
				if (dsq < best_dsq && dsq < my_range_sq) {
					best_enemy = j;
					best_dsq = dsq;
				}
			}
		}
	}

	// --- attack ---
	float my_cd = cooldown[i];

	if (best_enemy != 0xFFFFFFFFu && my_cd <= 0.0) {
		int dmg = int(atk_damage[i] * 100.0);
		atomicAdd(damage_acc[best_enemy], dmg);
		cooldown[i] = attack_cd_base;
		agent_info[i] = info | (1u << 9u);
	} else {
		cooldown[i] = max(my_cd - dt, 0.0);
		if (my_cd <= 0.0)
			agent_info[i] = info & ~(1u << 9u);
	}

	// --- death ---
	if (damage_acc[i] >= max_hp[i]) {
		agent_info[i] = agent_info[i] & ~1u;
	}
}
