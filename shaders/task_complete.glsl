#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { float pos_x[];    };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float pos_y[];    };
layout(set = 0, binding = 2, std430) restrict           buffer B2 { float a_stamina[];};
layout(set = 0, binding = 3, std430) restrict           buffer B3 { uint  a_target[]; };
layout(set = 0, binding = 4, std430) restrict readonly  buffer B4 { float t_pos_x[];  };
layout(set = 0, binding = 5, std430) restrict readonly  buffer B5 { float t_pos_y[];  };
layout(set = 0, binding = 6, std430) restrict readonly  buffer B6 { float t_cost[];   };
layout(set = 0, binding = 7, std430) restrict           buffer B7 { uint  t_status[]; };
layout(set = 0, binding = 8, std430) restrict readonly  buffer B8 { uint  t_type[];   };

layout(push_constant, std430) uniform Params {
	int   agent_count;
	int   task_count;
	float dt;
	float arrival_r_sq;
	float stamina_decay;
	int   _pad0;
	int   _pad1;
	int   _pad2;
};

const uint NONE = 0xFFFFFFFFu;

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (int(i) >= agent_count) return;

	float stam = a_stamina[i];
	stam = max(stam - stamina_decay * dt, 0.0);

	uint target = a_target[i];
	if (target != NONE) {
		float px = pos_x[i];
		float py = pos_y[i];
		float tx = t_pos_x[target];
		float ty = t_pos_y[target];
		float dx = px - tx;
		float dy = py - ty;
		float dsq = dx * dx + dy * dy;

		if (dsq < arrival_r_sq) {
			stam = clamp(stam - t_cost[target], 0.0, 1.0);
			t_status[target] = 3u;
			a_target[i] = NONE;
		}
	} else {
		if (stam < 0.3)
			stam = min(stam + 0.06 * dt, 0.3);
	}

	a_stamina[i] = stam;
}
