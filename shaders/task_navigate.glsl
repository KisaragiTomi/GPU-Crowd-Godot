#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0,  std430) restrict readonly  buffer B0  { float pos_x_in[];      };
layout(set = 0, binding = 1,  std430) restrict readonly  buffer B1  { float pos_y_in[];      };
layout(set = 0, binding = 2,  std430) restrict readonly  buffer B2  { float vel_x_in[];      };
layout(set = 0, binding = 3,  std430) restrict readonly  buffer B3  { float vel_y_in[];      };
layout(set = 0, binding = 4,  std430) restrict writeonly buffer B4  { float pos_x_out[];     };
layout(set = 0, binding = 5,  std430) restrict writeonly buffer B5  { float pos_y_out[];     };
layout(set = 0, binding = 6,  std430) restrict writeonly buffer B6  { float vel_x_out[];     };
layout(set = 0, binding = 7,  std430) restrict writeonly buffer B7  { float vel_y_out[];     };
layout(set = 0, binding = 8,  std430) restrict           buffer B8  { uint  a_target[];      };
layout(set = 0, binding = 9,  std430) restrict readonly  buffer B9  { float t_pos_x[];       };
layout(set = 0, binding = 10, std430) restrict readonly  buffer B10 { float t_pos_y[];       };
layout(set = 0, binding = 11, std430) restrict readonly  buffer B11 { float terrain[];       };
layout(set = 0, binding = 12, std430) restrict           buffer B12 { float stall[];         };
layout(set = 0, binding = 13, std430) restrict           buffer B13 { uint  t_status[];      };
layout(set = 0, binding = 14, std430) restrict readonly  buffer B14 { float goal_dist_all[]; };
layout(set = 0, binding = 15, std430) restrict readonly  buffer B15 { int   task_slot[];     };

layout(push_constant, std430) uniform Params {
	int   agent_count;   // 0
	float dt;            // 4
	float blend;         // 8
	float sep_r_sq;      // 12
	float sep_strength;  // 16
	float inv_sep_r;     // 20
	float world_w;       // 24
	float world_h;       // 28
	int   field_gw;      // 32
	int   field_gh;      // 36
	float field_cs;      // 40
	float inv_field_cs;  // 44
	uint  frame_seed;    // 48
	float nav_speed;     // 52
	int   cell_count;    // 56
	int   _pad1;         // 60
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

	float px = pos_x_in[i];
	float py = pos_y_in[i];
	float vx = vel_x_in[i];
	float vy = vel_y_in[i];
	float ifc = inv_field_cs;

	float dvx = 0.0, dvy = 0.0;
	uint target = a_target[i];

	if (target != 0xFFFFFFFFu) {
		int slot = task_slot[target];
		if (slot >= 0) {
			int cx = clamp(int(px * ifc), 1, field_gw - 2);
			int cy = clamp(int(py * ifc), 1, field_gh - 2);
			int base = slot * cell_count;
			int ci = cy * field_gw + cx;
			float ddx = goal_dist_all[base + ci + 1]
					  - goal_dist_all[base + ci - 1];
			float ddy = goal_dist_all[base + ci + field_gw]
					  - goal_dist_all[base + ci - field_gw];
			float glen = sqrt(ddx * ddx + ddy * ddy);
			if (glen > 1e-4) {
				dvx = -ddx / glen * nav_speed;
				dvy = -ddy / glen * nav_speed;
			}
			float tx = t_pos_x[target];
			float ty = t_pos_y[target];
			float tdist = sqrt((tx - px) * (tx - px) + (ty - py) * (ty - py));
			if (tdist < 40.0) {
				dvx *= tdist / 40.0;
				dvy *= tdist / 40.0;
			}
		} else {
			float tx = t_pos_x[target];
			float ty = t_pos_y[target];
			float dx = tx - px;
			float dy = ty - py;
			float dist = sqrt(dx * dx + dy * dy);
			if (dist > 0.5) {
				float spd = nav_speed;
				if (dist < 40.0) spd *= dist / 40.0;
				dvx = dx / dist * spd;
				dvy = dy / dist * spd;
			}
		}
	} else {
		float angle = rand01(frame_seed + i * 13u) * 6.28318;
		dvx = cos(angle) * 8.0;
		dvy = sin(angle) * 8.0;
	}

	float vsx = 0.0, vsy = 0.0;
	for (int j = 0; j < agent_count; j++) {
		if (uint(j) == i) continue;
		float djx = px - pos_x_in[j];
		float djy = py - pos_y_in[j];
		float dsq = djx * djx + djy * djy;
		if (dsq < sep_r_sq && dsq > 0.1) {
			float dd = sqrt(dsq);
			float f  = 1.0 - dd * inv_sep_r;
			float ff = sep_strength * f * f / dd;
			vsx += djx * ff;
			vsy += djy * ff;
		}
	}

	int tix = clamp(int(px * ifc), 1, field_gw - 2);
	int tiy = clamp(int(py * ifc), 1, field_gh - 2);
	int ti  = tiy * field_gw + tix;
	float wpx = 0.0, wpy = 0.0;
	if (terrain[ti + 1] > 0.5)        wpx -= 1.0;
	if (terrain[ti - 1] > 0.5)        wpx += 1.0;
	if (terrain[ti + field_gw] > 0.5)  wpy -= 1.0;
	if (terrain[ti - field_gw] > 0.5)  wpy += 1.0;
	dvx += wpx * 80.0;
	dvy += wpy * 80.0;

	vx += (dvx + vsx - vx) * blend;
	vy += (dvy + vsy - vy) * blend;

	float my_stall = stall[i];
	float spd = sqrt(vx * vx + vy * vy);
	if (spd < 5.0 && target != 0xFFFFFFFFu) {
		my_stall = min(my_stall + dt, 3.0);
	} else {
		my_stall = max(my_stall - dt * 2.0, 0.0);
	}
	if (my_stall > 0.5) {
		float esc = min(my_stall, 2.5) * 30.0;
		float angle = rand01(frame_seed / 20u + i) * 6.28318;
		vx += cos(angle) * esc;
		vy += sin(angle) * esc;
	}
	if (my_stall > 2.0 && target != 0xFFFFFFFFu) {
		t_status[target] = 0u;
		a_target[i] = 0xFFFFFFFFu;
	}
	stall[i] = my_stall;

	float npx = clamp(px + vx * dt, field_cs * 1.5, world_w - field_cs * 1.5);
	float npy = clamp(py + vy * dt, field_cs * 1.5, world_h - field_cs * 1.5);

	int ggx = clamp(int(npx * ifc), 0, field_gw - 1);
	int ggy = clamp(int(npy * ifc), 0, field_gh - 1);
	int gi  = ggy * field_gw + ggx;

	if (terrain[gi] > 0.5) {
		int gx2 = clamp(int(npx * ifc), 0, field_gw - 1);
		int gy2 = clamp(int(py  * ifc), 0, field_gh - 1);
		if (terrain[gy2 * field_gw + gx2] <= 0.5) {
			npy = py; vy *= 0.3;
		} else {
			gx2 = clamp(int(px  * ifc), 0, field_gw - 1);
			gy2 = clamp(int(npy * ifc), 0, field_gh - 1);
			if (terrain[gy2 * field_gw + gx2] <= 0.5) {
				npx = px; vx *= 0.3;
			} else {
				npx = px; npy = py; vx *= 0.5; vy *= 0.5;
			}
		}
	}

	pos_x_out[i] = npx;
	pos_y_out[i] = npy;
	vel_x_out[i] = vx;
	vel_y_out[i] = vy;
}
