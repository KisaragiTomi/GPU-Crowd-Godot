#[compute]
#version 450

// 调试显示（debug_stage 0..3）：把中间结果双线性重采样到 RGBA8 显示纹理，超出 [0, 1] 的值被截断
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 1, rgba8) writeonly uniform image2D output_image;
layout(set = 0, binding = 2, rgba16f) readonly uniform image2D emit_image;      // 0: emission
layout(set = 0, binding = 3, r32ui) readonly uniform uimage2D seed_image;       // 1: JFA 种子
layout(set = 0, binding = 4, r32f) readonly uniform image2D distance_image;     // 2: 距离场
layout(set = 0, binding = 5, rgba16f) readonly uniform image2D cascades_image;  // 3: cascade 紧凑存储

layout(set = 0, binding = 0, std430) restrict readonly buffer GIParamsBuffer {
    int   agent_count;
    float tex_w;
    float tex_h;
    float region_x;
    float region_y;
    float region_w;
    float region_h;
    float splat_radius;
    float emit_strength;
    int   terrain_w;
    int   terrain_h;
    float terrain_cell_size;
    float wall_alpha;
    float laser_width;
    float laser_emit_strength;
    float laser_life;
    int   laser_count;
    int   laser_cap;
    float interval_scale;
    float history_weight;
    float spatial_strength;
    int   num_cascades;
    int   merge_fix;
    int   debug_stage;
};

ivec2 input_size() {
    if (debug_stage == 0) return imageSize(emit_image);
    if (debug_stage == 1) return imageSize(seed_image);
    if (debug_stage == 2) return imageSize(distance_image);
    return imageSize(cascades_image);
}

vec4 load_input(ivec2 p) {
    if (debug_stage == 0) return imageLoad(emit_image, p);
    if (debug_stage == 1) {
        // 与原实现的种子图一致：种子像素为 (uv, 0, 1)，其余为 0
        uint seed = imageLoad(seed_image, p).r;
        if (seed == 0u) return vec4(0.0);
        return vec4(vec2(float(seed & 0xFFFFu), float(seed >> 16)) / vec2(imageSize(seed_image)), 0.0, 1.0);
    }
    if (debug_stage == 2) return vec4(vec3(imageLoad(distance_image, p).r), 1.0);
    return imageLoad(cascades_image, p);
}

void main() {
    // Get the global position of the current thread
    ivec2 output_pos = ivec2(gl_GlobalInvocationID.xy);

    // Check if we're within the output image bounds
    if (output_pos.x >= imageSize(output_image).x || output_pos.y >= imageSize(output_image).y) {
        return;
    }

    ivec2 in_size = input_size();
    // Calculate the scale factor
    vec2 scale = vec2(in_size) / vec2(imageSize(output_image));

    // Calculate the corresponding position in the input image
    vec2 input_pos_float = (vec2(output_pos) + 0.5) * scale - 0.5;
    ivec2 input_pos = ivec2(floor(input_pos_float));

    // Calculate the fractional part for interpolation
    vec2 frac = fract(input_pos_float);

    // Fetch the 4 neighboring pixels
    vec4 color00 = load_input(clamp(input_pos, ivec2(0), in_size - 1));
    vec4 color10 = load_input(clamp(input_pos + ivec2(1, 0), ivec2(0), in_size - 1));
    vec4 color01 = load_input(clamp(input_pos + ivec2(0, 1), ivec2(0), in_size - 1));
    vec4 color11 = load_input(clamp(input_pos + ivec2(1, 1), ivec2(0), in_size - 1));

    // Bilinear interpolation
    vec4 color = mix(
        mix(color00, color10, frac.x),
        mix(color01, color11, frac.x),
        frac.y
    );

    // Write the interpolated color to the output image
    imageStore(output_image, output_pos, clamp(color, 0.0, 1.0));
}
