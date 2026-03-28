#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { float density[];   };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float goal_dist[]; };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { float terrain[];   };
layout(set = 0, binding = 3, std430) restrict           buffer B3 { float fr[];        };
layout(set = 0, binding = 4, std430) restrict           buffer B4 { float fl[];        };
layout(set = 0, binding = 5, std430) restrict           buffer B5 { float fd[];        };
layout(set = 0, binding = 6, std430) restrict           buffer B6 { float fu[];        };

layout(push_constant, std430) uniform Params {
	int   gw;
	int   gh;
	float dt;
	float cell_size;
	float gravity;
	float damping;
	float density_scale;
	float goal_scale;
	float wall_scale;
	float max_flux;
	float _pad0;
	float _pad1;
};

void main() {
	int x = int(gl_GlobalInvocationID.x);
	int y = int(gl_GlobalInvocationID.y);
	if (x >= gw || y >= gh) return;

	int   i    = y * gw + x;
	float ds   = density_scale;
	float gs   = goal_scale;
	float ws   = wall_scale;
	float H    = density[i] * ds + goal_dist[i] * gs + terrain[i] * ws;
	float coef = dt * gravity / cell_size;
	float dmp  = damping;
	float mf   = max_flux;

	// Right
	if (x < gw - 1) {
		int j = i + 1;
		float Hj = density[j] * ds + goal_dist[j] * gs + terrain[j] * ws;
		fr[i] = clamp(dmp * fr[i] + coef * (H - Hj), 0.0, mf);
	} else {
		fr[i] = 0.0;
	}

	// Left
	if (x > 0) {
		int j = i - 1;
		float Hj = density[j] * ds + goal_dist[j] * gs + terrain[j] * ws;
		fl[i] = clamp(dmp * fl[i] + coef * (H - Hj), 0.0, mf);
	} else {
		fl[i] = 0.0;
	}

	// Down
	if (y < gh - 1) {
		int j = i + gw;
		float Hj = density[j] * ds + goal_dist[j] * gs + terrain[j] * ws;
		fd[i] = clamp(dmp * fd[i] + coef * (H - Hj), 0.0, mf);
	} else {
		fd[i] = 0.0;
	}

	// Up
	if (y > 0) {
		int j = i - gw;
		float Hj = density[j] * ds + goal_dist[j] * gs + terrain[j] * ws;
		fu[i] = clamp(dmp * fu[i] + coef * (H - Hj), 0.0, mf);
	} else {
		fu[i] = 0.0;
	}
}
