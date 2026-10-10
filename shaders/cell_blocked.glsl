#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { uint  cell_attacker_in[]; };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { uint  agent_info[];       };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { uint  fac_to_group[];     };
layout(set = 0, binding = 3, std430) restrict readonly  buffer B3 { float terrain[];          };
layout(set = 0, binding = 4, std430) restrict writeonly buffer B4 { uint  cell_blocked_out[];  };

layout(push_constant, std430) uniform Params {
	int field_gw;
	int field_gh;
	int agent_count;
	int _pad;
};

void main() {
	int fx = int(gl_GlobalInvocationID.x);
	int fy = int(gl_GlobalInvocationID.y);
	if (fx >= field_gw || fy >= field_gh) return;

	int gi = fy * field_gw + fx;
	uint blocked = 0u;

	if (terrain[gi] > 0.5) {
		blocked = 0xFFFFFFFFu;
	} else {
		uint occ = cell_attacker_in[gi];
		if (occ != 0xFFFFFFFFu && int(occ) < agent_count) {
			uint occ_info = agent_info[occ];
			if ((occ_info & 1u) != 0u) {
				uint occ_fac = (occ_info >> 1u) & 0x1Fu;
				uint occ_grp = fac_to_group[occ_fac];
				blocked = ~(1u << occ_grp);
			}
		}
	}

	cell_blocked_out[gi] = blocked;
}
