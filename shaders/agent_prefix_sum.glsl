#[compute]
#version 450

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { uint cell_count[]; };
layout(set = 0, binding = 1, std430) restrict writeonly buffer B1 { uint cell_start[]; };
layout(set = 0, binding = 2, std430) restrict writeonly buffer B2 { uint cell_offset[]; };

layout(push_constant, std430) uniform Params {
	int num_cells;
	int _p0; int _p1; int _p2;
};

void main() {
	uint sum = 0u;
	for (int i = 0; i < num_cells; i++) {
		cell_start[i]  = sum;
		cell_offset[i] = sum;
		sum += cell_count[i];
	}
}
