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
layout(set = 0, binding = 8,  std430) restrict readonly  buffer B8  { uint  cs_buf[];     };
layout(set = 0, binding = 9,  std430) restrict readonly  buffer B9  { uint  cc_buf[];     };
layout(set = 0, binding = 10, std430) restrict readonly  buffer B10 { uint  si_buf[];     };
layout(set = 0, binding = 11, std430) restrict readonly  buffer B11 { float out_vx[];     };
layout(set = 0, binding = 12, std430) restrict readonly  buffer B12 { float out_vy[];     };
layout(set = 0, binding = 13, std430) restrict readonly  buffer B13 { float terrain[];    };
layout(set = 0, binding = 14, std430) restrict readonly  buffer B14 { float goal_dist[];  };
layout(set = 0, binding = 15, std430) restrict           buffer B15 { float stall[];      };

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

	float px = pos_x_in[i];
	float py = pos_y_in[i];
	float vx = vel_x_in[i];
	float vy = vel_y_in[i];

	// --- bilinear velocity sample ---
	float ifc = inv_field_cs;
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
	float vfx = out_vx[i00]*w00 + out_vx[i10]*w10 + out_vx[i01]*w01 + out_vx[i11]*w11;
	float vfy = out_vy[i00]*w00 + out_vy[i10]*w10 + out_vy[i01]*w01 + out_vy[i11]*w11;

	// --- neighbor separation via cell buffer ---
	float vsx = 0.0, vsy = 0.0;
	int n_count = 0;
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
				float djx = px - pos_x_in[j];
				float djy = py - pos_y_in[j];
				float dsq = djx * djx + djy * djy;
				if (dsq < sep_r_sq && dsq > 0.1) {
					float dd = sqrt(dsq);
					float f  = 1.0 - dd * inv_sep_r;
					float ff = sep_strength * f * f / dd;
					vsx += djx * ff;
					vsy += djy * ff;
					n_count++;
				}
			}
		}
	}

	// --- wall avoidance force (prevent piling up against walls) ---
	int tix = clamp(int(px * ifc), 1, field_gw - 2);
	int tiy = clamp(int(py * ifc), 1, field_gh - 2);
	int ti  = tiy * field_gw + tix;
	float wpx = 0.0, wpy = 0.0;
	if (terrain[ti + 1] > 0.5)        wpx -= 1.0;
	if (terrain[ti - 1] > 0.5)        wpx += 1.0;
	if (terrain[ti + field_gw] > 0.5)  wpy -= 1.0;
	if (terrain[ti - field_gw] > 0.5)  wpy += 1.0;
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

	// --- stall detection + tangential escape ---
	float my_stall = stall[i];
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
	stall[i] = my_stall;

	float npx = clamp(px + vx * dt, field_cs * 1.5, world_w - field_cs * 1.5);
	float npy = clamp(py + vy * dt, field_cs * 1.5, world_h - field_cs * 1.5);

	int ggx = clamp(int(npx * ifc), 0, field_gw - 1);
	int ggy = clamp(int(npy * ifc), 0, field_gh - 1);
	int gi  = ggy * field_gw + ggx;

	if (terrain[gi] > 0.5) {
		// Wall sliding: try X-only, then Y-only, then full stop
		int gx2 = clamp(int(npx * ifc), 0, field_gw - 1);
		int gy2 = clamp(int(py  * ifc), 0, field_gh - 1);
		if (terrain[gy2 * field_gw + gx2] <= 0.5) {
			npy = py; vy *= 0.1;
			gi = gy2 * field_gw + gx2;
		} else {
			gx2 = clamp(int(px  * ifc), 0, field_gw - 1);
			gy2 = clamp(int(npy * ifc), 0, field_gh - 1);
			if (terrain[gy2 * field_gw + gx2] <= 0.5) {
				npx = px; vx *= 0.1;
				gi = gy2 * field_gw + gx2;
			} else {
				npx = px; npy = py; vx *= 0.3; vy *= 0.3;
			}
		}
	}

	if (goal_dist[gi] < 2.0) {
		uint seed = frame_seed + i * 3u;
		npx = mix(field_cs * 2.0, field_cs * 12.0, rand01(seed));
		npy = mix(field_cs * 3.0, world_h - field_cs * 3.0, rand01(seed + 1u));
		vx = 0.0; vy = 0.0;
	}

	pos_x_out[i] = npx;  pos_y_out[i] = npy;
	vel_x_out[i] = vx;   vel_y_out[i] = vy;
}
