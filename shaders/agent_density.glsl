#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { float pos_x[]; };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float pos_y[]; };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { uint cell_start_buf[]; };
layout(set = 0, binding = 3, std430) restrict readonly  buffer B3 { uint cell_count_buf[]; };
layout(set = 0, binding = 4, std430) restrict readonly  buffer B4 { uint sorted_idx[]; };
layout(set = 0, binding = 5, std430) restrict writeonly buffer B5 { float density_out[]; };
layout(set = 0, binding = 6, std430) restrict readonly  buffer B6 { uint agent_info[]; };
layout(set = 0, binding = 7, std430) restrict writeonly buffer B7 { uint faction_presence_out[]; };

layout(push_constant, std430) uniform Params {
	int   field_gw;
	int   field_gh;
	float field_cs;
	float inv_field_cs;
	int   sg_w;
	int   sg_h;
	float sg_cs;
	int   _p0;
};

void main() {
	int fx = int(gl_GlobalInvocationID.x);
	int fy = int(gl_GlobalInvocationID.y);
	if (fx >= field_gw || fy >= field_gh) return;

	float ifc = inv_field_cs;
	float fc  = field_cs;
	float d   = 0.0;
	uint  presence = 0u;

	int sx_lo = max(0,       int(float(max(fx - 1, 0))           * fc / sg_cs));
	int sx_hi = min(sg_w - 1, int(float(min(fx + 2, field_gw)) * fc / sg_cs));
	int sy_lo = max(0,       int(float(max(fy - 1, 0))           * fc / sg_cs));
	int sy_hi = min(sg_h - 1, int(float(min(fy + 2, field_gh)) * fc / sg_cs));

	for (int sy = sy_lo; sy <= sy_hi; sy++) {
		for (int sx = sx_lo; sx <= sx_hi; sx++) {
			int ci    = sy * sg_w + sx;
			int start = int(cell_start_buf[ci]);
			int cnt   = int(cell_count_buf[ci]);
			for (int k = 0; k < cnt; k++) {
				uint aid = sorted_idx[start + k];

				uint ainfo = agent_info[aid];
				if ((ainfo & 1u) == 0u) continue;
				uint afac = (ainfo >> 1u) & 0x1Fu;
				presence |= (1u << afac);

				float gx = pos_x[aid] * ifc;
				float gy = pos_y[aid] * ifc;
				int ix = int(floor(gx));
				int iy = int(floor(gy));
				float frac_x = gx - float(ix);
				float frac_y = gy - float(iy);

				if      (ix == fx     && iy == fy)     d += (1.0 - frac_x) * (1.0 - frac_y);
				else if (ix + 1 == fx && iy == fy)     d += frac_x * (1.0 - frac_y);
				else if (ix == fx     && iy + 1 == fy) d += (1.0 - frac_x) * frac_y;
				else if (ix + 1 == fx && iy + 1 == fy) d += frac_x * frac_y;
			}
		}
	}
	int gi = fy * field_gw + fx;
	density_out[gi] = d;
	faction_presence_out[gi] = presence;
}
