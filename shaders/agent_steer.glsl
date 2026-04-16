#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0,  std430) restrict readonly  buffer B0  { float pos_x_in[];  };
layout(set = 0, binding = 1,  std430) restrict readonly  buffer B1  { float pos_y_in[];  };
layout(set = 0, binding = 2,  std430) restrict readonly  buffer B2  { float vel_x_in[];  };
layout(set = 0, binding = 3,  std430) restrict readonly  buffer B3  { float vel_y_in[];  };
layout(set = 0, binding = 4,  std430) restrict writeonly buffer B4  { float pos_x_out[]; };
layout(set = 0, binding = 5,  std430) restrict writeonly buffer B5  { float pos_y_out[]; };
layout(set = 0, binding = 6,  std430) restrict writeonly buffer B6  { float vel_x_out[]; };
layout(set = 0, binding = 7,  std430) restrict writeonly buffer B7  { float vel_y_out[]; };
layout(set = 0, binding = 8,  std430) restrict readonly  buffer B8  { uint  cs_buf[];    };
layout(set = 0, binding = 9,  std430) restrict readonly  buffer B9  { uint  cc_buf[];    };
layout(set = 0, binding = 10, std430) restrict readonly  buffer B10 { uint  si_buf[];    };
layout(set = 0, binding = 11, std430) restrict readonly  buffer B11 { float terrain[];   };
layout(set = 0, binding = 12, std430) restrict           buffer B12 { float stall[];     };

layout(set = 1, binding = 0, std430) restrict          buffer S1B0 { uint  agent_info[];    };
layout(set = 1, binding = 1, std430) restrict readonly buffer S1B1 { float out_vx_all[];   };
layout(set = 1, binding = 2, std430) restrict readonly buffer S1B2 { float out_vy_all[];   };
layout(set = 1, binding = 3, std430) restrict readonly buffer S1B3 { uint  fac_to_group[]; };
layout(set = 1, binding = 4, std430) restrict          buffer S1B4 { uint  corpse_map[];   };
layout(set = 1, binding = 5, std430) restrict          buffer S1B5 { uint  cell_attacker[]; };
layout(set = 1, binding = 6, std430) restrict          buffer S1B6 { float cooldown_buf[];  };

layout(push_constant, std430) uniform Params {
	int   agent_count;    // 0
	int   sg_w;           // 4
	int   sg_h;           // 8
	float sg_inv_cs;      // 12
	float dt;             // 16
	float blend;          // 20
	float sep_r_sq;       // 24
	float inv_sep_r;      // 28
	float sep_strength;   // 32
	float inv_field_cs;   // 36
	int   field_gw;       // 40
	int   field_gh;       // 44
	float world_w;        // 48
	float world_h;        // 52
	float field_cs;       // 56
	uint  frame_seed;     // 60
};

uint pcg(uint v) {
	uint s = v * 747796405u + 2891336453u;
	uint w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
	return (w >> 22u) ^ w;
}

float rand01(uint seed) { return float(pcg(seed)) / 4294967295.0; }

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (int(i) >= agent_count) return;

	uint info = agent_info[i];
	if ((info & 1u) == 0u) {
		// --- corpse placement: find nearest empty cell, no overlap ---
		float px = pos_x_in[i];
		float py = pos_y_in[i];
		int cx = clamp(int(px * inv_field_cs), 0, field_gw - 1);
		int cy = clamp(int(py * inv_field_cs), 0, field_gh - 1);

		if (corpse_map[cy * field_gw + cx] == i) {
			pos_x_out[i] = px; pos_y_out[i] = py;
			vel_x_out[i] = 0.0; vel_y_out[i] = 0.0;
			return;
		}

		bool placed = false;
		for (int r = 0; r <= 10 && !placed; r++) {
			for (int dy = -r; dy <= r && !placed; dy++) {
				for (int dx = -r; dx <= r && !placed; dx++) {
					if (max(abs(dx), abs(dy)) != r) continue;
					int nx = cx + dx;
					int ny = cy + dy;
					if (nx < 0 || nx >= field_gw || ny < 0 || ny >= field_gh) continue;
					int ni = ny * field_gw + nx;
					if (terrain[ni] > 0.5) continue;
					uint prev = atomicCompSwap(corpse_map[ni], 0xFFFFFFFFu, i);
					if (prev == 0xFFFFFFFFu) {
						px = (float(nx) + 0.5) * field_cs;
						py = (float(ny) + 0.5) * field_cs;
						placed = true;
					}
				}
			}
		}

		pos_x_out[i] = px; pos_y_out[i] = py;
		vel_x_out[i] = 0.0; vel_y_out[i] = 0.0;
		return;
	}

	float px = pos_x_in[i];
	float py = pos_y_in[i];
	float vx = vel_x_in[i];
	float vy = vel_y_in[i];
	float ifc = inv_field_cs;

	// --- goal-group velocity sampling ---
	uint faction = (info >> 1u) & 0x1Fu;
	uint group   = fac_to_group[faction];
	int  vbase   = int(group) * field_gw * field_gh;

	float sgx = clamp(px * ifc, 0.5, float(field_gw) - 1.5);
	float sgy = clamp(py * ifc, 0.5, float(field_gh) - 1.5);
	int six = clamp(int(sgx), 0, field_gw - 2);
	int siy = clamp(int(sgy), 0, field_gh - 2);
	float sfx = sgx - float(six);
	float sfy = sgy - float(siy);
	int i00 = siy * field_gw + six;
	int i10 = i00 + 1;
	int i01 = i00 + field_gw;
	int i11 = i01 + 1;
	float w00 = (1.0 - sfx) * (1.0 - sfy);
	float w10 = sfx * (1.0 - sfy);
	float w01 = (1.0 - sfx) * sfy;
	float w11 = sfx * sfy;
	float vfx = out_vx_all[vbase+i00]*w00 + out_vx_all[vbase+i10]*w10 + out_vx_all[vbase+i01]*w01 + out_vx_all[vbase+i11]*w11;
	float vfy = out_vy_all[vbase+i00]*w00 + out_vy_all[vbase+i10]*w10 + out_vy_all[vbase+i01]*w01 + out_vy_all[vbase+i11]*w11;

	// --- neighbor separation via cell buffer ---
	float vsx = 0.0, vsy = 0.0;
	int n_count = 0;
	float atk_impede = 0.0;
	int hcx = clamp(int(px * sg_inv_cs), 0, sg_w - 1);
	int hcy = clamp(int(py * sg_inv_cs), 0, sg_h - 1);
	for (int ny = max(hcy - 1, 0); ny <= min(hcy + 1, sg_h - 1); ny++) {
		int row = ny * sg_w;
		for (int nx = max(hcx - 1, 0); nx <= min(hcx + 1, sg_w - 1); nx++) {
			int ci  = row + nx;
			int s   = int(cs_buf[ci]);
			int cnt = int(cc_buf[ci]);
			for (int k = 0; k < cnt; k++) {
				uint j = si_buf[s + k];
				if (j == i) continue;
				if ((agent_info[j] & 1u) == 0u) continue;
				float djx = px - pos_x_in[j];
				float djy = py - pos_y_in[j];
				float dsq = djx * djx + djy * djy;
				if (dsq < sep_r_sq && dsq > 0.1) {
					float dd = sqrt(dsq);
					float f  = 1.0 - dd * inv_sep_r;
					float ff = sep_strength * f * f / dd;

					if ((info & (1u << 9u)) == 0u) {
						uint jv = agent_info[j];
						if ((jv & (1u << 9u)) != 0u) {
							uint jg = fac_to_group[(jv >> 1u) & 0x1Fu];
							if (jg != group) {
								ff *= 3.0;
								atk_impede = max(atk_impede, f);
							} else {
								ff *= 2.0;
								atk_impede = max(atk_impede, f * 0.5);
							}
						}
					}

					vsx += djx * ff;
					vsy += djy * ff;
					n_count++;
				}
			}
		}
	}

	// --- wall avoidance force ---
	int tix = clamp(int(px * ifc), 1, field_gw - 2);
	int tiy = clamp(int(py * ifc), 1, field_gh - 2);
	int ti  = tiy * field_gw + tix;
	float wpx = 0.0, wpy = 0.0;
	if (terrain[ti + 1] > 0.5)        wpx -= 1.0;
	if (terrain[ti - 1] > 0.5)        wpx += 1.0;
	if (terrain[ti + field_gw] > 0.5)  wpy -= 1.0;
	if (terrain[ti - field_gw] > 0.5)  wpy += 1.0;

	// --- friendly attacker cell avoidance (non-attackers only) ---
	if ((info & (1u << 9u)) == 0u) {
		for (int dy = -1; dy <= 1; dy++) {
			int eny = tiy + dy;
			if (eny < 0 || eny >= field_gh) continue;
			for (int dx = -1; dx <= 1; dx++) {
				if (dx == 0 && dy == 0) continue;
				int enx = tix + dx;
				if (enx < 0 || enx >= field_gw) continue;
				int eci = eny * field_gw + enx;
				uint ca = cell_attacker[eci];
				if (ca != 0xFFFFFFFFu && ca < uint(agent_count)) {
					uint ca_info = agent_info[ca];
					if ((ca_info & 1u) != 0u && (ca_info & (1u << 9u)) != 0u) {
						uint ca_group = fac_to_group[(ca_info >> 1u) & 0x1Fu];
						if (ca_group == group) {
							wpx -= float(dx);
							wpy -= float(dy);
						}
					}
				}
			}
		}
	}

	float wall_repel = 40.0;
	vfx += wpx * wall_repel;
	vfy += wpy * wall_repel;

	// --- high-density noise to break deadlock symmetry ---
	if (n_count > 6) {
		float noise_amp = float(n_count - 6) * 3.0;
		vsx += (rand01(frame_seed + i * 7u)     - 0.5) * noise_amp;
		vsy += (rand01(frame_seed + i * 7u + 1u) - 0.5) * noise_amp;
	}

	// --- steer + integrate ---
	vx += (vfx + vsx - vx) * blend;
	vy += (vfy + vsy - vy) * blend;

	if (atk_impede > 0.0 && (info & (1u << 9u)) == 0u) {
		float slow = 1.0 - atk_impede * 0.7;
		vx *= slow;
		vy *= slow;
	}

	// --- attack freeze (displaced agents keep moving) ---
	bool is_attacking = (info & (1u << 9u)) != 0u;
	bool is_displaced = (info & (1u << 10u)) != 0u;
	if (is_attacking) {
		vx = 0.0; vy = 0.0;
	}

	// --- stall detection + tangential escape (skip for attacking agents) ---
	float my_stall = stall[i];
	if (!is_attacking) {
		float spd = sqrt(vx * vx + vy * vy);
		if (spd < 5.0 && n_count > 3) {
			my_stall = min(my_stall + dt, 3.0);
		} else {
			my_stall = max(my_stall - dt * 3.0, 0.0);
		}
		if (my_stall > 0.4) {
			float esc = min(my_stall, 2.5) * 18.0;
			float flen = sqrt(vfx * vfx + vfy * vfy);
			if (flen > 0.1) {
				float sign = (rand01(frame_seed / 20u + i) > 0.5) ? 1.0 : -1.0;
				vx += (-vfy / flen) * sign * esc;
				vy += ( vfx / flen) * sign * esc;
			} else {
				float angle = rand01(frame_seed / 20u + i) * 6.28318;
				vx += cos(angle) * esc;
				vy += sin(angle) * esc;
			}
		}
	} else {
		my_stall = 0.0;
	}
	stall[i] = my_stall;

	float npx = clamp(px + vx * dt, field_cs * 1.5, world_w - field_cs * 1.5);
	float npy = clamp(py + vy * dt, field_cs * 1.5, world_h - field_cs * 1.5);

	int ggx = clamp(int(npx * ifc), 0, field_gw - 1);
	int ggy = clamp(int(npy * ifc), 0, field_gh - 1);
	int gi  = ggy * field_gw + ggx;

	if (terrain[gi] > 0.5) {
		int ogx = clamp(int(px * ifc), 0, field_gw - 1);
		int ogy = clamp(int(py * ifc), 0, field_gh - 1);
		if (terrain[ogy * field_gw + ogx] > 0.5) {
			// Agent already stuck in wall — gradually push towards nearest open cell
			float bd = 1e10;
			float bx = px, by = py;
			for (int ey = -3; ey <= 3; ey++) {
				for (int ex = -3; ex <= 3; ex++) {
					int enx = clamp(ogx + ex, 1, field_gw - 2);
					int eny = clamp(ogy + ey, 1, field_gh - 2);
					if (terrain[eny * field_gw + enx] < 0.5) {
						float cx = (float(enx) + 0.5) * field_cs;
						float cy = (float(eny) + 0.5) * field_cs;
						float dd = (cx - px) * (cx - px) + (cy - py) * (cy - py);
						if (dd < bd) { bd = dd; bx = cx; by = cy; }
					}
				}
			}
			float tox = bx - px;
			float toy = by - py;
			float tod = sqrt(tox * tox + toy * toy);
			float esc_step = min(tod, 120.0 * dt);
			if (tod > 0.01) {
				npx = px + (tox / tod) * esc_step;
				npy = py + (toy / tod) * esc_step;
			}
			vx = 0.0; vy = 0.0;
		} else {
			// Normal: moving from open space into wall — axis separation
			int gx2 = clamp(int(npx * ifc), 0, field_gw - 1);
			int gy2 = clamp(int(py  * ifc), 0, field_gh - 1);
			if (terrain[gy2 * field_gw + gx2] <= 0.5) {
				npy = py; vy *= 0.1;
			} else {
				gx2 = clamp(int(px  * ifc), 0, field_gw - 1);
				gy2 = clamp(int(npy * ifc), 0, field_gh - 1);
				if (terrain[gy2 * field_gw + gx2] <= 0.5) {
					npx = px; vx *= 0.1;
				} else {
					npx = px; npy = py; vx *= 0.3; vy *= 0.3;
				}
			}
		}
	}

	// --- hard minimum distance projection (faction-aware) ---
	float safe_x = npx, safe_y = npy;
	bool i_attacking = (info & (1u << 9u)) != 0u;
	{
		float min_dist    = field_cs * 0.75;
		float full_dist   = field_cs;
		int pcx = clamp(int(npx * sg_inv_cs), 0, sg_w - 1);
		int pcy = clamp(int(npy * sg_inv_cs), 0, sg_h - 1);
		for (int pny = max(pcy - 1, 0); pny <= min(pcy + 1, sg_h - 1); pny++) {
			int prow = pny * sg_w;
			for (int pnx = max(pcx - 1, 0); pnx <= min(pcx + 1, sg_w - 1); pnx++) {
				int pci  = prow + pnx;
				int ps   = int(cs_buf[pci]);
				int pcnt = int(cc_buf[pci]);
				for (int pk = 0; pk < pcnt; pk++) {
					uint j2 = si_buf[ps + pk];
					if (j2 == i) continue;
					uint j2_info = agent_info[j2];
					if ((j2_info & 1u) == 0u) continue;
					bool j_atk    = (j2_info & (1u << 9u)) != 0u;
					uint j_group  = fac_to_group[(j2_info >> 1u) & 0x1Fu];
					bool same_team = (group == j_group);

				if (i_attacking) {
					if (!j_atk) continue;
					bool j_disp = (j2_info & (1u << 10u)) != 0u;
					if (!is_displaced && j_disp) continue;
					if (is_displaced && same_team) continue;
				}

					float ed, push_frac;
					if (i_attacking) {
						ed = full_dist;  push_frac = 0.5;
					} else if (j_atk && !same_team) {
						ed = full_dist * 1.5;  push_frac = 1.0;
					} else if (j_atk) {
						ed = full_dist;  push_frac = 0.5;
					} else if (!same_team) {
						ed = full_dist;  push_frac = 1.0;
					} else {
						ed = min_dist;   push_frac = 0.5;
					}

					float dx = npx - pos_x_in[j2];
					float dy = npy - pos_y_in[j2];
					float d2 = dx * dx + dy * dy;
					float ed_sq = ed * ed;
					if (d2 < ed_sq && d2 > 0.01) {
						float d = sqrt(d2);
						float push = (ed - d) * push_frac;
						npx += (dx / d) * push;
						npy += (dy / d) * push;
					}
				}
			}
		}
	}
	npx = clamp(npx, field_cs * 1.5, world_w - field_cs * 1.5);
	npy = clamp(npy, field_cs * 1.5, world_h - field_cs * 1.5);

	// Revert projection if it pushed into terrain (only if safe pos is open)
	int fgx = clamp(int(npx * ifc), 0, field_gw - 1);
	int fgy = clamp(int(npy * ifc), 0, field_gh - 1);
	if (terrain[fgy * field_gw + fgx] > 0.5) {
		int sgx = clamp(int(safe_x * ifc), 0, field_gw - 1);
		int sgy = clamp(int(safe_y * ifc), 0, field_gh - 1);
		if (terrain[sgy * field_gw + sgx] <= 0.5) {
			npx = safe_x;
			npy = safe_y;
		}
	}

	// --- cell conflict resolution for attacking agents ---
	if (is_attacking) {
		int atk_cx = clamp(int(npx * ifc), 0, field_gw - 1);
		int atk_cy = clamp(int(npy * ifc), 0, field_gh - 1);
		int atk_ci = atk_cy * field_gw + atk_cx;

		uint prev = atomicCompSwap(cell_attacker[atk_ci], 0xFFFFFFFFu, i);

		if (prev == 0xFFFFFFFFu || prev == i) {
			if (is_displaced) {
				agent_info[i] = info & ~(1u << 10u);
				cooldown_buf[i] = 0.0;
			}
			float tgt_x = (float(atk_cx) + 0.5) * field_cs;
			float tgt_y = (float(atk_cy) + 0.5) * field_cs;
			float smx = tgt_x - npx;
			float smy = tgt_y - npy;
			float smd = sqrt(smx * smx + smy * smy);
		if (smd > 0.1) {
			float sstep = min(80.0 * dt, smd);
			npx += (smx / smd) * sstep;
			npy += (smy / smd) * sstep;
			} else {
				npx = tgt_x;
				npy = tgt_y;
			}
		} else {
			if (!is_displaced) {
				agent_info[i] = info | (1u << 10u);
			}

			float best_dd = 1e10;
			int best_nx = atk_cx, best_ny = atk_cy;
			bool found_cell = false;
			for (int sr = 1; sr <= 5 && !found_cell; sr++) {
				for (int dy = -sr; dy <= sr; dy++) {
					for (int dx = -sr; dx <= sr; dx++) {
						if (max(abs(dx), abs(dy)) != sr) continue;
						int enx = atk_cx + dx;
						int eny = atk_cy + dy;
						if (enx < 0 || enx >= field_gw || eny < 0 || eny >= field_gh) continue;
						if (terrain[eny * field_gw + enx] > 0.5) continue;
						int eci = eny * field_gw + enx;
						uint ecv = cell_attacker[eci];
						if (ecv != 0xFFFFFFFFu && ecv != i) continue;
						float ecx = (float(enx) + 0.5) * field_cs;
						float ecy = (float(eny) + 0.5) * field_cs;
						float dd = (ecx - px) * (ecx - px) + (ecy - py) * (ecy - py);
						if (dd < best_dd) {
							best_dd = dd;
							best_nx = enx;
							best_ny = eny;
							found_cell = true;
						}
					}
				}
			}

			if (found_cell) {
				float tx = (float(best_nx) + 0.5) * field_cs;
				float ty = (float(best_ny) + 0.5) * field_cs;
				float mx = tx - px;
				float my = ty - py;
				float md = sqrt(mx * mx + my * my);
				float disp_speed = 120.0;
				if (md > 0.01) {
					float step = min(disp_speed * dt, md);
					npx = px + (mx / md) * step;
					npy = py + (my / md) * step;
				}
			} else {
				float sep_d = sqrt(vsx * vsx + vsy * vsy);
				float disp_speed = 120.0;
				if (sep_d > 0.1) {
					npx = px + (vsx / sep_d) * disp_speed * dt;
					npy = py + (vsy / sep_d) * disp_speed * dt;
				} else {
					float angle = rand01(i * 137u + 77u) * 6.28318;
					npx = px + cos(angle) * disp_speed * dt;
					npy = py + sin(angle) * disp_speed * dt;
				}
			}

			int check_gx = clamp(int(npx * ifc), 0, field_gw - 1);
			int check_gy = clamp(int(npy * ifc), 0, field_gh - 1);
			if (terrain[check_gy * field_gw + check_gx] > 0.5) {
				npx = px;
				npy = py;
			}
		}
	}

	if (is_attacking && !is_displaced) {
		float max_disp = 80.0 * dt;
		float ddx = npx - px;
		float ddy = npy - py;
		float ddd = sqrt(ddx * ddx + ddy * ddy);
		if (ddd > max_disp) {
			npx = px + (ddx / ddd) * max_disp;
			npy = py + (ddy / ddd) * max_disp;
		}
	}

	npx = clamp(npx, field_cs * 1.5, world_w - field_cs * 1.5);
	npy = clamp(npy, field_cs * 1.5, world_h - field_cs * 1.5);

	pos_x_out[i] = npx;  pos_y_out[i] = npy;
	vel_x_out[i] = vx;   vel_y_out[i] = vy;
}
