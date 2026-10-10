#[compute]
#version 450

layout(local_size_x=16,local_size_y=16)in;

// 种子 = 像素坐标 x | y << 16，0 表示无种子（gi_seed.glsl）
layout(set=0,binding=1,r32ui)writeonly uniform uimage2D output_image;
layout(set=0,binding=2,r32ui)readonly uniform uimage2D input_image;

layout(set = 0, binding = 4, std430) restrict readonly buffer ParamsBuffer {
    float uOffset;
};

const float FLT_MAX = 3.4028235e38;

const vec2 offsets[9] = vec2[9](
    vec2(-1.0, -1.0),
    vec2(-1.0, 0.0),
    vec2(-1.0, 1.0),
    vec2(0.0, -1.0),
    vec2(0.0, 0.0),
    vec2(0.0, 1.0),
    vec2(1.0, -1.0),
    vec2(1.0, 0.0),
    vec2(1.0, 1.0)
);


void main(){
    vec2 fragCoord=gl_GlobalInvocationID.xy;
    ivec2 ifragCoord=ivec2(fragCoord.xy);
    ivec2 input_image_size=imageSize(input_image).xy;
    if (ifragCoord.x >= input_image_size.x || ifragCoord.y >= input_image_size.y) return;

    float closest_dist = FLT_MAX;
    uint closest_seed = 0u;

    for(int i = 0; i < 9; i++) {
        ivec2 jump = ifragCoord + ivec2(offsets[i] * uOffset);
        if (any(lessThan(jump, ivec2(0))) || any(greaterThanEqual(jump, input_image_size))) {
            continue;
        }
        uint seed = imageLoad(input_image, jump).r;
        if (seed == 0u) continue;
        // 仍按原实现的 uv（= 坐标 / 尺寸）× 尺寸计算，种子相同时距离逐位一致（JFA 平局的取舍也不变）
        vec2 seedpos = vec2(float(seed & 0xFFFFu), float(seed >> 16)) / vec2(input_image_size);
        vec2 diff =(seedpos * input_image_size) - fragCoord;
        float dist=dot(diff,diff);

        if (dist < closest_dist) {
            closest_dist = dist;
            closest_seed = seed;
        }
    }

    imageStore(output_image, ifragCoord, uvec4(closest_seed));

}
