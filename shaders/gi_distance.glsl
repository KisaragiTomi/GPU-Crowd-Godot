#[compute]
#version 450

layout(local_size_x=16,local_size_y=16)in;

layout(set=0,binding=1,r32f)writeonly uniform image2D output_image;
// JFA 结果：最近种子的像素坐标 x | y << 16；0 表示没找到（与原实现一样按 (0,0) 算距离）
layout(set=0,binding=2,r32ui)readonly uniform uimage2D input_image;

void main(){
    vec2 fragCoord=gl_GlobalInvocationID.xy;
    ivec2 ifragCoord=ivec2(fragCoord.xy);
    ivec2 input_image_size=imageSize(input_image).xy;
    if (ifragCoord.x >= input_image_size.x || ifragCoord.y >= input_image_size.y) return;

    uint seed = imageLoad(input_image, ifragCoord).r;
    // 与原实现相同的 uv × 尺寸算法，距离逐位一致
    vec2 nearestSeed = vec2(float(seed & 0xFFFFu), float(seed >> 16)) / vec2(input_image_size);
    float distance = distance(fragCoord, nearestSeed * input_image_size);
    imageStore(output_image,ifragCoord,vec4(distance));
}
