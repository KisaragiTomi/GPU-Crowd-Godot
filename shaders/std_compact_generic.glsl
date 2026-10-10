#[compute]
#version 450

layout(local_size_x = 8, local_size_y = 8, local_size_z = 1) in;

layout(set = 0, binding = 0, std430) restrict readonly  buffer B0 { float activity[]; };
layout(set = 0, binding = 1, std430) restrict readonly  buffer B1 { float mask_buf[]; };
layout(set = 0, binding = 2, std430) restrict           buffer B2 { uint  compact_counter; };
layout(set = 0, binding = 3, std430) restrict writeonly  buffer B3 { uint  compact_tile_coords[]; };

layout(push_constant, std430) uniform Params {
	int   gw;
	int   gh;
	int   expand_pixels;
	float threshold;
};

shared uint gs_tile_active;

void main() {
	uint li = gl_LocalInvocationIndex;
	ivec2 tile_coord = ivec2(gl_WorkGroupID.xy);
	ivec2 tile_origin = tile_coord * ivec2(8, 8);
	ivec2 max_cell = ivec2(gw, gh);

	if (tile_origin.x >= max_cell.x || tile_origin.y >= max_cell.y) return;

	if (li == 0u) gs_tile_active = 0u;
	barrier();

	ivec2 scan_min = max(tile_origin - ivec2(expand_pixels), ivec2(0));
	ivec2 scan_max = min(tile_origin + ivec2(8) + ivec2(expand_pixels) - 1, max_cell - 1);
	ivec2 scan_size = scan_max - scan_min + ivec2(1);
	uint total = uint(scan_size.x) * uint(scan_size.y);

	for (uint i = li; i < total; i += 64u) {
		if (gs_tile_active != 0u) break;

		int px = int(scan_min.x) + int(i) % scan_size.x;
		int py = int(scan_min.y) + int(i) / scan_size.x;
		int idx = py * gw + px;

		if (mask_buf[idx] < 0.5 && activity[idx] > threshold)
			atomicOr(gs_tile_active, 1u);
	}

	barrier();

	if (li == 0u && gs_tile_active != 0u) {
		uint slot = atomicAdd(compact_counter, 1u);
		compact_tile_coords[slot] = uint(tile_coord.x) | (uint(tile_coord.y) << 16u);
	}
}
