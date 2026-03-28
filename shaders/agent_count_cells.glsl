#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { float pos_x[]; };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float pos_y[]; };
layout(set = 0, binding = 2, std430) restrict           buffer B2 { uint cell_count[]; };

layout(push_constant, std430) uniform Params {
	int   agent_count;
	int   sg_w;
	int   sg_h;
	float sg_inv_cs;
};

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (int(i) >= agent_count) return;
	int cx = clamp(int(pos_x[i] * sg_inv_cs), 0, sg_w - 1);
	int cy = clamp(int(pos_y[i] * sg_inv_cs), 0, sg_h - 1);
	atomicAdd(cell_count[cy * sg_w + cx], 1u);
}
