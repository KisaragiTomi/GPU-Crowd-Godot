#[compute]
#version 450

layout(local_size_x = 16, local_size_y = 16) in;

layout(set = 0, binding = 0, rgba32f) readonly uniform image2D current_image;
layout(set = 0, binding = 1, rgba32f) readonly uniform image2D history_image;
layout(set = 0, binding = 2, rgba32f) writeonly uniform image2D output_image;

layout(set = 0, binding = 3, std430) restrict readonly buffer ParamsBuffer {
	float history_weight;
	float spatial_strength;
	int _pad0;
	int _pad1;
};

vec3 read_current(ivec2 pos, ivec2 sz) {
	return imageLoad(current_image, clamp(pos, ivec2(0), sz - ivec2(1))).rgb;
}

void main() {
	ivec2 pos = ivec2(gl_GlobalInvocationID.xy);
	ivec2 sz = imageSize(output_image);
	if (pos.x >= sz.x || pos.y >= sz.y) return;

	vec3 center = read_current(pos, sz);
	vec3 accum = center * 4.0;
	accum += read_current(pos + ivec2(-1, 0), sz) * 2.0;
	accum += read_current(pos + ivec2(1, 0), sz) * 2.0;
	accum += read_current(pos + ivec2(0, -1), sz) * 2.0;
	accum += read_current(pos + ivec2(0, 1), sz) * 2.0;
	accum += read_current(pos + ivec2(-1, -1), sz);
	accum += read_current(pos + ivec2(1, -1), sz);
	accum += read_current(pos + ivec2(-1, 1), sz);
	accum += read_current(pos + ivec2(1, 1), sz);

	vec3 blurred = accum / 16.0;
	vec3 filtered = mix(center, blurred, clamp(spatial_strength, 0.0, 1.0));
	vec3 history = imageLoad(history_image, pos).rgb;
	vec3 stable = mix(filtered, history, clamp(history_weight, 0.0, 0.98));

	imageStore(output_image, pos, vec4(stable, 1.0));
}
