#[compute]
#version 450

layout(local_size_x = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { uint compact_counter; };
layout(set = 0, binding = 1, std430) restrict writeonly  buffer B1 { uint indirect_args[3]; };

layout(push_constant, std430) uniform Params {
	int _dummy0;
	int _dummy1;
	int _dummy2;
	int _dummy3;
};

void main() {
	uint tile_count = compact_counter;
	indirect_args[0] = max(tile_count, 1u);
	indirect_args[1] = 1u;
	indirect_args[2] = 1u;
}
