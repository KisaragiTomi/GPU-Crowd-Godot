#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0,  std430) restrict readonly  buffer B0  { float a_pos_x[];   };
layout(set = 0, binding = 1,  std430) restrict readonly  buffer B1  { float a_pos_y[];   };
layout(set = 0, binding = 2,  std430) restrict readonly  buffer B2  { float a_stamina[]; };
layout(set = 0, binding = 3,  std430) restrict readonly  buffer B3  { uint  a_target[];  };
layout(set = 0, binding = 4,  std430) restrict readonly  buffer B4  { float t_pos_x[];   };
layout(set = 0, binding = 5,  std430) restrict readonly  buffer B5  { float t_pos_y[];   };
layout(set = 0, binding = 6,  std430) restrict readonly  buffer B6  { float t_cost[];    };
layout(set = 0, binding = 7,  std430) restrict readonly  buffer B7  { float t_priority[];};
layout(set = 0, binding = 8,  std430) restrict readonly  buffer B8  { uint  t_status[];  };
layout(set = 0, binding = 9,  std430) restrict readonly  buffer B9  { uint  t_type[];    };
layout(set = 0, binding = 10, std430) restrict writeonly buffer B10 { uint  best_task[]; };
layout(set = 0, binding = 11, std430) restrict writeonly buffer B11 { uint  best_bid[];  };
layout(set = 0, binding = 12, std430) restrict           buffer B12 { uint  t_claim[];   };

layout(push_constant, std430) uniform Params {
	int agent_count;
	int task_count;
	int _pad0;
	int _pad1;
};

const uint  NONE       = 0xFFFFFFFFu;
const float TIRED      = 0.28;
const float VERY_TIRED = 0.14;

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (int(i) >= agent_count) return;

	if (a_target[i] != NONE) {
		best_task[i] = NONE;
		best_bid[i]  = NONE;
		return;
	}

	float ax   = a_pos_x[i];
	float ay   = a_pos_y[i];
	float stam = a_stamina[i];

	float max_score = -1.0;
	uint  max_j     = NONE;
	float max_dist  = 1e6;

	for (int j = 0; j < task_count; j++) {
		if (t_status[j] != 0u) continue;

		float tx = t_pos_x[j];
		float ty = t_pos_y[j];
		float dx = ax - tx;
		float dy = ay - ty;
		float dist = sqrt(dx * dx + dy * dy);
		float cost = t_cost[j];

		bool feasible = (cost <= 0.0) || (stam > cost + 0.05);
		if (!feasible) continue;

		float dist_score = 1.0 / (1.0 + dist * 0.02);

		float urgency = 1.0;
		uint  ttype   = t_type[j];

		if (ttype >= 1u) {
			if (stam < VERY_TIRED)
				urgency = 10.0 + (VERY_TIRED - stam) * 80.0;
			else if (stam < TIRED)
				urgency = 3.0 + (TIRED - stam) * 30.0;
			else
				urgency = 0.3;
		} else {
			if (stam < VERY_TIRED)
				urgency = 0.05;
			else if (stam < TIRED)
				urgency = 0.4;
		}

		float score = t_priority[j] * dist_score * urgency;

		if (score > max_score) {
			max_score = score;
			max_j     = uint(j);
			max_dist  = dist;
		}
	}

	best_task[i] = max_j;

	if (max_j != NONE) {
		uint dist_enc = min(uint(max_dist), 65535u);
		uint bid      = (dist_enc << 16u) | (i & 0xFFFFu);
		best_bid[i]   = bid;
		atomicMin(t_claim[max_j], bid);
	} else {
		best_bid[i] = NONE;
	}
}
