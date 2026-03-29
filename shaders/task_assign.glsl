#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { uint best_task[]; };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { uint best_bid[];  };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { uint t_claim[];   };
layout(set = 0, binding = 3, std430) restrict           buffer B3 { uint a_target[];  };
layout(set = 0, binding = 4, std430) restrict           buffer B4 { uint t_status[];  };
layout(set = 0, binding = 5, std430) restrict writeonly buffer B5 { uint t_owner[];   };

layout(push_constant, std430) uniform Params {
	int agent_count;
	int task_count;
	int _pad0;
	int _pad1;
};

const uint NONE = 0xFFFFFFFFu;

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (int(i) >= agent_count) return;

	if (a_target[i] != NONE) return;

	uint j = best_task[i];
	if (j == NONE) return;

	if (t_claim[j] == best_bid[i]) {
		a_target[i] = j;
		t_status[j] = 1u;
		t_owner[j]  = i;
	}
}
