# FluidCrowd Memory

## Ground movement cost

2026-10-10: 新增独立 `ground_cost` / `buf_ground_cost`，整数 1–4，默认 1。`swe_velocity.glsl` 的空闲和拥挤分支都将目标速度除以该代价。`goal_bfs.glsl` 同时松弛原来的未加权 `bfs_dist` 和新 `bfs_cost`；`goal_dist_all` 继续用于战斗距离门控，`goal_dist` / gdir 使用加权移动代价，避免地面难度缩短射程。加权区域使用单侧下降方向并确定性处理等价绕行，普通邻域保留原梯度。动态场仍每帧轮转一个组、预算 80 次 pass；复杂迷宫可能超预算，开局按批迭代直到收敛。

操作：4 或 Ground 按钮选地面画刷，滑块 1–4 倍，左键绘制、右键恢复 1；3 擦除墙和难度，R 重新生成噪声地面和城堡。地面用 RF 纹理着色，Sprite 在主控背景之上、墙和单位之下；不参与碰撞或 GI 遮挡。暂停也能绘制，恢复后各组最多 32 个模拟帧刷新导航。

2026-10-10 后续：用户要求地面类型按噪声散布。`crowd_sim._generate_ground()` 在城堡生成后、静态数据上传前运行（启动和重置共用），采用 FastNoiseLite Simplex Smooth + 3 层 FBM，频率 0.022，阈值 0.04 / 0.26 / 0.50 分为 1–4 倍代价；跳过墙和地图边界。导出属性 `ground_noise_seed` 默认 0，每次随机，非零可复现；`ground_noise_frequency` 越小斑块越大。全场景验证覆盖四类型、相邻格成片、固定种子复现、CPU/GPU/纹理一致及随机重置，预览图 `screenshots/ground-noise.png`。

验证：Godot 4.6.1 shader 导入与脚本解析通过；`tests/ground_cost_test.gd` 在 RTX 5070 Vulkan 运行，GPU 距离对照独立 Dijkstra（普通/绕行/走廊/不可达/随机地面），检查下降方向、射程距离不变、两组四档速度（空闲/拥挤）和画刷操作，0 失败。完整 20k 场景检查暂停绘制上传、40 个运行帧、重置后的 CPU/GPU/显示纹理恢复，退出无错误。预览图在忽略目录 `screenshots/ground-difficulty-live.png`。

## GI Integration

当前工程已接入从 `radiance-cascades-godot/try-mix-jason-and-shadertoy` 复制来的 2D Radiance Cascades GI 方案。

| File | Role |
| --- | --- |
| `scripts/gpu_gi.gd` | `GPUGI` 主控：emission（原子累加）→ JFA / distance → 按 cascade 逐级派发的 RC → image → stabilize；uniform set 预建 / 缓存 |
| `scripts/crowd_sim.gd` | 创建 `GPUGI`，每隔 `_gi_frame_skip` 帧把当前 agent buffer、紧凑激光列表及条数 `_laser_count` 输入 GI，并显示 `_gi_sprite`（RGBA8 `ImageTexture`） |
| `scripts/gpu_agents.gd` | 保存远程攻击 per-agent laser line/ttl transient buffers；`laser_compact` 把可见激光压缩成紧凑列表 `buf_laser_compact`，屏幕绘制和 GI emission 共用 |
| `shaders/agent_combat.glsl` | 远程单位实际攻击时写入短生命周期 laser line/ttl |
| `shaders/laser_compact.glsl` | 把 ttl > 0 且长度² ≥ 1 的激光压缩成紧凑列表（最多 `LASER_CAP` = 8192 条），供 CPU 一次回读后批量绘制，GI 也直接读它 |
| `shaders/gi_wall_mask.glsl` | 清空 emission 累加缓冲并写入墙体 alpha（原 `gi_clear` + `gi_wall_mask` 合并的逐像素 pass） |
| `shaders/gi_agent_emit.glsl` | 一次 dispatch 画全部发光体：前 `agent_count` 个线程画 agent 光斑，之后每条激光 16 个线程分段画；对 fp32 位模式 `atomicMax`（原 `gi_laser_emit` 已并入） |
| `shaders/gi_seed.glsl` | 累加缓冲 → RGBA16F `tex_emit`，同时生成 R32UI JFA 种子（`x \| y << 16`，0 = 无） |
| `shaders/gi_distance.glsl` | 生成 R32F 像素单位距离场，供 RC raymarch 使用 |
| `shaders/gi_jump_flood_algorithm.glsl` | Jump Flood Algorithm（R32UI ping-pong） |
| `shaders/gi_rc_cube_a.glsl` | 复制来的 Radiance Cascades cascade 计算 pass，每次 dispatch 只算一级 |
| `shaders/gi_rc_image.glsl` | 复制来的 Radiance Cascades final image pass（cascade 0 → tonemap 后的 GI） |
| `shaders/gi_stabilize.glsl` | 3×3 空间滤波 + temporal history（RGBA16F），同时写 RGBA8 显示纹理 |
| `shaders/gi_resample_image.glsl` | `debug_stage` 0–3 的中间结果重采样到 RGBA8 显示纹理 |

## Runtime Notes

- `GPUGI.compute_gi()` 流程：clear + walls -> emit（agent + laser，原子累加）-> resolve + seed -> JFA -> distance -> cascade 5..0（每级一次 dispatch，级间 barrier）-> final GI image -> stabilize。RGBA8 显示纹理 `tex_display` 在 GI 这次提交里用 `texture_get_data_async` 登记回读，随 `sync()` 到达；`get_output_image()` 返回 `FORMAT_RGBA8`，缺数据时退回同步 `texture_get_data`。
- 纹理格式：`tex_emit` / cascades / `tex_output` / history ×2 为 RGBA16F，JFA 种子 R32UI，距离场 R32F，显示纹理 RGBA8；另有每像素 `uvec4`（fp32 位模式）的 emission 累加 storage buffer 供 `atomicMax`。800×480 时 GI 显存约 55 MB（原约 140 MB）。history 必须保持浮点：8 bit 会让指数衰减卡在最低档留下残影。
- Cascade 布局（`gi_rc_cube_a.glsl` 的 GetC0Size / GetCascadeSize / GetCascadeLinearOffset，`GPUGI._compute_cascade_layout()` 用同一公式算派发大小）：c0 探针 (512, 512·h/w)、4 方向，第 i 级探针每轴 >> i、方向 ×4^i；5:3 视口下 6 级共 3,713,024 像素，与 GI 精度无关。所有级按线性下标紧凑存放在一张宽 `CASCADE_TEX_W` = 2048 的纹理里（5:3 时 2048×1813），shader 里 `idx & 2047` / `idx >> 11` 寻址。按级从高到低派发后每级读到的都是本次更新刚写完的上一级（原来 6 个 1024² 面跨面读写有竞争，单次结果依赖上一次 GI，是拖影来源之一）。线程分配只影响速度：≥ 2 级按"同方向、相邻探针"排、0–1 级按存储顺序。
- GI 参数：一个 96 字节的 `params_buf` 每次更新 `buffer_update` 一次（布局见 `_update_params` 注释，各 shader 的 ParamsBuffer 取其前缀）；JFA 跳距、cascade 级号 / 像素数是各 pass 的静态小 buffer。GI shader 仍不用 push constants（沿用约定）。所有 uniform set 在 `_init` 预建；emission 那个按 agent / 激光缓冲 RID 缓存（位置双缓冲，通常两份），外部缓冲被释放（Godot 会连带释放 set）时自动重建；`cleanup()` 先释放 set 再释放纹理 / 缓冲。
- JFA 种子存整数坐标，但距离仍按原实现的 `uv = 坐标 / 尺寸`、`uv × 尺寸` 计算，种子相同时距离场逐位一致（JFA 平局取舍也不变）。
- 激光 GI 读 `GPUAgents.buf_laser_compact`（主提交里 `laser_compact.glsl` 已写好）和 `crowd_sim._laser_count`，条数截到 `LASER_CAP` = 8192，超出的不发光（100k 实测最多约 2000 条）。与原规则（per-agent、要求存活且远程）的差别只有"本帧战斗里刚死、ttl 尚未清零"的激光，实测约占 0.03%。
- `debug_stage` 默认是 `4`，含义是 final GI；`0` 可看 emission map，`1` 看 JFA 种子，`2` 看 distance，`3` 看 cascade 紧凑存储纹理（全部级）；0–3 写进 RGBA8 显示纹理，超出 [0, 1] 截断。
- L 键重新打开 GI 时把 `gi.temporal_valid` 置 false，不再与关闭前的过时 history 混合。
- GI 纹理尺寸 = 视口尺寸 × `_gi_precision`（UI 滑块 0.5–1.0，默认 0.5；1600×960 视口下为 800×480），覆盖相机视野 × `GI_REGION_SCALE` 的区域，显示时由 `_gi_sprite` 的 position/scale 贴回世界坐标。
- 每个 live agent 通过 `gi_agent_emit.glsl` 产生平滑圆形 emission，颜色来自 `agents.buf_mm_fac_colors`。
- Combat 命中会写入 `GPUAgents.buf_hit_flash`，`gi_agent_emit.glsl` 读取它让受击单位的阵营色 GI 短暂增强并快速衰减。
- GI 显示强度在 `crowd_sim.gd`：`GI_AGENT_EMIT_STRENGTH = 1.15`、`GI_AGENT_SPLAT_RADIUS_SCALE = 1.0`、`GI_DISPLAY_ALPHA = 0.60`。
- 为避免相机 zoom 改变 GI 亮度，agent splat radius 和 laser GI width 都按世界单位定义，在每次 GI update 前用当前 `region.size / tex_size` 换算成 texel；`gi_agent_emit.glsl` 对小于 1 texel 的 agent 光斑和激光都做 subpixel coverage 衰减。
- `GPUGI.interval_scale` 会同步缩放 `gi_rc_cube_a.glsl` / `gi_rc_image.glsl` 的 C0 ray interval，让 GI 传播距离尽量保持世界尺度一致，而不是随屏幕 zoom 漂移。
- 远程攻击激光参数在 `crowd_sim.gd`：`RANGED_LASER_LIFE = 0.12`、`RANGED_LASER_GI_WIDTH_WORLD = 0.45`、`RANGED_LASER_GI_STRENGTH = 3.6`。屏幕上的短暂柔边细线由 `laser_compact.glsl` 的紧凑列表驱动：soft / glow / core 三个 additive `LaserOverlay` 图层（宽 `1.55/0.78/0.28`，`self_modulate.a` = `0.055/0.16/0.62`）各调用一次 `draw_multiline_colors`，顶点 alpha = fade。

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

2026-10-01: 性能优化中与 GI 相关的部分。屏幕激光改为 `laser_compact.glsl` 紧凑列表 + 三个图层各一次 `draw_multiline_colors`，GI 仍读 per-agent 的 `laser_line` / `laser_ttl`，GI 各 pass 本身未改。GI 输出纹理改为 `texture_get_data_async` 随 GI 提交回读；每帧的 MultiMesh / 计数 / 激光回读也改为 `buffer_get_data_async` 随主提交到达。原来每次 `buffer_get_data` / `texture_get_data` 都临时分配主机可见内存再 flush+stall，100k agent 时会偶发数百 ms 的卡顿。
Verification: 全部脚本 `--check-only` 通过，`godot --headless --import` 无 shader 错误，非 headless `--quit-after 1500` 无新报错；异步与同步回读逐字节比较（mm、计数、激光、GI stage 0–4、调试叠加、选中查询）全部一致。

2026-10-02: GI 性能优化（第二批）。Radiance cascades 改为按级派发、每级只覆盖本级像素，所有级紧凑存放在一张 2048 宽的 RGBA16F 纹理里（去掉 6 个 1024² RGBA32F 面、面 4/5 与面 3 后半段的空像素派发，以及跨面读写竞争）；emission RGBA16F、距离场 R32F、JFA 种子 R32UI；clear + wall mask 合成一个 pass；agent 光斑与激光合成一次 dispatch（激光改读紧凑列表，每条 16 线程分段画），写入改为 fp32 位模式 `atomicMax`，不再丢更新；stabilize 另写 RGBA8 显示纹理，回读 / `ImageTexture` 上传从 6 MB 降到 1.5 MB；uniform set 预建缓存；`gi_clear.glsl`、`gi_laser_emit.glsl` 和未被引用的 `gi_radiance_cascades.glsl` 已删除。RTX 5070、800×480 实测：GI GPU 每次更新约 3.46 → 2.0 ms（cascade 3.09 → 1.80 ms，激光 0.12 → 合并后 emission 共 0.03 ms），录制命令 0.24–0.32 → 0.05–0.07 ms，`ImageTexture.update` 1.3 → 0.33 ms（p50），GI 显存 139.8 → 54.6 MB。
Verification: `godot --headless --import` 后全部 shader 编译无错，`gpu_gi.gd` / `crowd_sim.gd` `--check-only` 通过。同一帧输入对比 HEAD 管线（同一份 emission 上连续跑到收敛）：注入同一份 emission 时距离场逐位一致，GI（tonemap 后 0..255）MAE ≈ 0.04、最大 0.21（fp16 引起）；端到端 MAE 0.24（20k）/ 0.41（100k），差异集中在原实现竞争丢更新的像素上（新版 emission 与 CPU 双精度精确结果在 1e-3 内一致，原实现每次约丢 1700 / 3600 像素）。原管线单次结果相对收敛值 MAE 1.2–1.7、镜头平移 32 世界单位时 2.8–3.1，新管线单次即收敛。独立本地设备上反复创建 / 运行 / cleanup 后 `rd.free()` 无 RID 泄漏报告；真实运行中切换 GI 精度 0.75 → 1.0 → 0.5、L 键关开 GI、debug_stage 0–4 轮换均无报错。

2026-10-02: 退出时 `crowd_sim._exit_tree` 依次释放 GI、叠加层、agents、field 的 RID，最后 `rd.free()` 本地 RenderingDevice（原来从不释放，每次退出都报 ObjectDB 泄漏，并偶发退出段错误）。各模块 `cleanup()` 释放 uniform set 前先检查 `uniform_set_is_valid`：跨模块引用的缓冲先被释放时，Godot 会连带释放引用它的 set。
Verification: 连续 6 次 `--quit-after 300` 退出码均为 0，无 invalid ID 报错，RenderingDevice 泄漏消失；剩下偶发的 ObjectDB 泄漏只有背景音乐的 `AudioStreamMP3` 播放实例（Dummy 音频驱动），与 GPU 代码无关。
