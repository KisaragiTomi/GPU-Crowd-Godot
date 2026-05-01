# FluidCrowd Memory

## GI Integration

当前工程已接入从 `radiance-cascades-godot/try-mix-jason-and-shadertoy` 复制来的 2D Radiance Cascades GI 方案。

| File | Role |
| --- | --- |
| `scripts/gpu_gi.gd` | `GPUGI` 主控：生成 agent emission map，执行 JFA/distance/RC/image pass |
| `scripts/crowd_sim.gd` | 创建 `GPUGI`，每隔 `_gi_frame_skip` 帧把当前 agent / laser buffer 输入 GI，并显示 `_gi_sprite` |
| `scripts/gpu_agents.gd` | 保存远程攻击 laser line/ttl transient buffers，供屏幕绘制和 GI emission 共用 |
| `shaders/agent_combat.glsl` | 远程单位实际攻击时写入短生命周期 laser line/ttl |
| `shaders/gi_agent_emit.glsl` | 每个 live agent 按 faction color splat 到 `tex_emit` |
| `shaders/gi_laser_emit.glsl` | 把远程攻击激光线段 stamp 到 `tex_emit`，作为 GI 输入 |
| `shaders/gi_seed.glsl` | 从 emission alpha 生成 JFA seed |
| `shaders/gi_distance.glsl` | 生成像素单位距离场，供 RC raymarch 使用 |
| `shaders/gi_jump_flood_algorithm.glsl` | Jump Flood Algorithm |
| `shaders/gi_rc_cube_a.glsl` | 复制来的 Radiance Cascades cascade 计算 pass |
| `shaders/gi_rc_image.glsl` | 复制来的 Radiance Cascades final image pass |

## Runtime Notes

- `GPUGI.compute_gi()` 流程：clear emission -> wall mask -> agent emit -> laser emit -> seed -> JFA -> distance -> six cascade faces -> final GI image -> stabilize.
- 新 GI shader 参数使用 small storage buffer，不使用 push constants，避免 Godot 4.6 对新增 shader 的 push constant 反射问题。
- `debug_stage` 默认是 `4`，含义是 final GI；`0` 可看 emission map，`1` 看 JFA，`2` 看 distance，`3` 看 cascade face。
- 当前 `_gi_tex_w = 400`、`_gi_tex_h = 240`，显示时由 `_gi_sprite.scale` 拉伸到 `world_size`。
- 每个 live agent 通过 `gi_agent_emit.glsl` 产生平滑圆形 emission，颜色来自 `agents.buf_mm_fac_colors`。
- Combat 命中会写入 `GPUAgents.buf_hit_flash`，`gi_agent_emit.glsl` 读取它让受击单位的阵营色 GI 短暂增强并快速衰减。
- GI 显示强度在 `crowd_sim.gd`：`GI_AGENT_EMIT_STRENGTH = 1.15`、`GI_AGENT_SPLAT_RADIUS_SCALE = 1.0`、`GI_DISPLAY_ALPHA = 0.60`。
- 为避免相机 zoom 改变 GI 亮度，agent splat radius 和 laser GI width 都按世界单位定义，在每次 GI update 前用当前 `region.size / tex_size` 换算成 texel；`gi_agent_emit.glsl` 和 `gi_laser_emit.glsl` 对小于 1 texel 的发光体做 subpixel coverage 衰减。
- `GPUGI.interval_scale` 会同步缩放 `gi_rc_cube_a.glsl` / `gi_rc_image.glsl` 的 C0 ray interval，让 GI 传播距离尽量保持世界尺度一致，而不是随屏幕 zoom 漂移。
- 远程攻击激光参数在 `crowd_sim.gd`：`RANGED_LASER_LIFE = 0.12`、`RANGED_LASER_GI_WIDTH_WORLD = 0.45`、`RANGED_LASER_GI_STRENGTH = 3.6`；同一 buffer 也驱动 additive `LaserOverlay` 的短暂柔边细线绘制。

## Verification

已用 Godot 4.6.1 执行：

```bash
godot --headless --path d:/MyProject/FluidCrowd --check-only --script res://scripts/gpu_gi.gd
godot --headless --path d:/MyProject/FluidCrowd --check-only --script res://scripts/crowd_sim.gd
godot --headless --path d:/MyProject/FluidCrowd --import
godot --path d:/MyProject/FluidCrowd --quit-after 8
```

结果：脚本解析通过，shader 导入通过，实际启动无 GI 运行时错误。`user://gi_debug_stage4.png` 和 `user://gi_viewport.png` 非空，能看到 agent emission 产生的 GI 光晕。

2026-05-01 追加验证：相机 zoom GI 归一化修改后，已再次执行 `crowd_sim.gd --check-only`、`gpu_gi.gd --check-only`、`godot --headless --import`、`godot --headless --quit-after 5`，均返回成功。

2026-05-01: Attack laser tuning narrowed the screen overlay to `RANGED_LASER_VIS_WIDTH = 0.55` plus a 1.15 faction-colored glow, narrowed GI laser world width to `0.65`, and changed `gi_laser_emit.glsl` to emit exact source faction color instead of warming it toward white.
Verification: `crowd_sim.gd --check-only`, `gpu_gi.gd --check-only`, `godot --headless --import`, and non-headless `godot --quit-after 5` succeeded. Headless runtime still hits the existing Vulkan RenderingDevice assertion before simulation init.

2026-05-01: Strengthened GI after visual review: agent emission now uses `GI_AGENT_EMIT_STRENGTH = 2.4`, source radius scale `1.35`, display alpha `0.85`, and laser GI strength `6.5`.
Verification: `crowd_sim.gd --check-only`, `gpu_gi.gd --check-only`, `godot --headless --import`, and non-headless `godot --quit-after 5` succeeded after the GI strength tuning.

2026-05-01: Added hit GI flash feedback. `agent_combat.glsl` writes a fixed-point `hit_flash` value on the damaged target and decays it quickly each frame; `gi_agent_emit.glsl` boosts same-faction emission color/radius while flash is active. Verified `crowd_sim.gd`, `gpu_agents.gd`, `gpu_gi.gd` check-only, `godot --headless --import`, and non-headless `godot --quit-after 5`.

2026-05-01: Reduced character GI after ghosting review: agent emission `1.15`, radius scale `1.0`, display alpha `0.60`, hit flash value `0.75` with decay `12.0`. Attack laser presentation moved to additive `LaserOverlay`, using thinner core/glow/soft widths `0.28/0.78/1.55`; laser GI width/strength reduced to `0.45/3.6`.
Verification: `crowd_sim.gd --check-only`, `gpu_gi.gd --check-only`, `godot --headless --import`, and non-headless `godot --quit-after 5` succeeded after the ghosting/additive laser tuning.
