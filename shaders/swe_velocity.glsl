#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { float fr[];      };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float fl[];      };
layout(set = 0, binding = 2, std430) restrict readonly  buffer B2 { float fd[];      };
layout(set = 0, binding = 3, std430) restrict readonly  buffer B3 { float fu[];      };
layout(set = 0, binding = 4, std430) restrict readonly  buffer B4 { float terrain[]; };
layout(set = 0, binding = 5, std430) restrict readonly  buffer B5 { float density[]; };
layout(set = 0, binding = 6, std430) restrict readonly  buffer B6 { float gdir_x[];  };
layout(set = 0, binding = 7, std430) restrict readonly  buffer B7 { float gdir_y[];  };
layout(set = 0, binding = 8, std430) restrict writeonly buffer B8 { float out_vx[];  };
layout(set = 0, binding = 9, std430) restrict writeonly buffer B9 { float out_vy[];  };

layout(push_constant, std430) uniform Params {
	int   gw;
	int   gh;
	float agent_speed;
	float density_slowdown;
};

void main() {
	int x = int(gl_GlobalInvocationID.x);
	int y = int(gl_GlobalInvocationID.y);
	if (x >= gw || y >= gh) return;

	int i = y * gw + x;

	if (terrain[i] > 0.5) {
		out_vx[i] = 0.0;
		out_vy[i] = 0.0;
		return;
	}

	float fx   = fr[i] - fl[i];
	float fy   = fd[i] - fu[i];
	float fmag = sqrt(fx * fx + fy * fy);

	float dir_x, dir_y;
	if (fmag > 0.001) {
		dir_x = fx / fmag;
		dir_y = fy / fmag;
	} else {
		dir_x = gdir_x[i];
		dir_y = gdir_y[i];
	}

	float local_speed = agent_speed / (1.0 + density[i] * density_slowdown);
	out_vx[i] = dir_x * local_speed;
	out_vy[i] = dir_y * local_speed;
}
