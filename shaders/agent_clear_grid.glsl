#[compute]
#version 450

layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict writeonly buffer B0 { uint cell_count[]; };

layout(push_constant, std430) uniform Params {
	int num_cells;
	int _p0; int _p1; int _p2;
};

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (int(i) >= num_cells) return;
	cell_count[i] = 0u;
}
