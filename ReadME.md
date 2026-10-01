# FluidCrowd

GPU 加速的流体驱动群集模拟，基于 Godot 4.6 + Vulkan Compute Shader 实现。

核心思路：用浅水方程（SWE）生成连续速度场，驱动上万 agent 自然流动、避障、聚散，同时支持多阵营战斗。

## 项目概览

| 项 | 值 |
|----|-----|
| 引擎 | Godot 4.6（自编译，Vulkan 后端） |
| 语言 | GDScript + GLSL 450 Compute Shader |
| Agent 上限 | 100,000 |
| 阵营 | 32 个阵营，32 个目标组 |
| 网格 | 800 × 480，cell_size = 8px |
| 数据布局 | SoA（Structure of Arrays） |
| 城堡模板 | 黑白 PNG 图转换为地形墙壁 |

## 架构

```text
crowd_sim.gd                主控：初始化、仿真循环、UI、输入、渲染
├── gpu_swe_field.gd        流体场：SWE flux/velocity、BFS 目标场、地形
├── gpu_agents.gd           Agent GPU 管理：buffer、dispatch、MultiMesh、激光压缩、选中查询
├── gpu_gi.gd               GI：2D Radiance Cascades（见 MEMORY.md）
└── sparse_tile_dispatch.gd 稀疏 tile 调度：活跃区域压缩 + indirect dispatch（当前未启用）
```

## GPU 管线（每帧）

主模拟在一次 `compute_list_begin/end`、一次 `submit/sync` 中按序 dispatch：

| 顺序 | Pass | 内容 |
| --- | --- | --- |
| 1 | Spatial Hash Build | `clear_grid → count_cells → prefix_sum → scatter` |
| 2 | Density | 双线性 splatting → `density_out` + `faction_presence` |
| 3 | SWE Flux | 节流约 33 ms 一次；共享内存加载 H 场 → 四方向 flux 更新 |
| 4 | Velocity Field | 32 个目标组；flux 差分 + 目标梯度 → 每 cell 速度向量 |
| 5 | Combat | 距离场门控 → 空间哈希邻居扫描 → 攻击/受伤/死亡 |
| 5.5 | Cell Blocked | 攻击者占格 → 各 group 的阻挡掩码 |
| 6 | Steer + Integrate | 目标速度 + 分离力 + 墙壁回避 + 拥堵逃逸；被挤开时按需求逃逸方向；地形碰撞 + 硬距离投影 |
| 7 | Build MultiMesh | 位置 + 阵营颜色 → MultiMesh 缓冲布局（Transform2D + 颜色） |
| 7.5 | Laser Compact | 可见激光 → 紧凑列表（最多 8192 条） |
| 7.6 | Select Inspect | 仅选中 agent 时：最近敌人 / blocker 查询 + 阻挡格叠加 |
| 7.7 | Goal BFS | 每帧 1 个目标组（轮转）：init → 分块松弛 → 梯度 |

- 各阶段之间由 `compute_list_add_barrier` 同步。
- 需要回读的数据（MultiMesh 缓冲、计数、激光列表、选中查询）在 `compute_list_end` 之后用 `buffer_get_data_async` 登记，随同一次 `sync()` 到达，不额外 flush/stall。
- 主模拟之后：GI 每 2 帧单独提交一次（见 `MEMORY.md`）；density/velocity 调试叠加（可选）单独提交。

## 文件结构

```text
FluidCrowd/
├── scenes/
│   └── main.tscn                  主场景（CrowdSim + Camera2D）
├── scripts/
│   ├── crowd_sim.gd               主控脚本
│   ├── gpu_agents.gd              Agent GPU 管理
│   ├── gpu_swe_field.gd           SWE 流体场 + BFS 目标场
│   ├── gpu_gi.gd                  GI（Radiance Cascades）
│   └── sparse_tile_dispatch.gd    稀疏 tile 调度模块（当前未启用）
├── shaders/
│   ├── agent_clear_grid.glsl      空间哈希：清零
│   ├── agent_count_cells.glsl     空间哈希：原子计数
│   ├── agent_prefix_sum.glsl      空间哈希：前缀和（单个 1024 线程工作组）
│   ├── agent_scatter.glsl         空间哈希：索引散射
│   ├── agent_density.glsl         密度 + 阵营 presence 计算
│   ├── agent_combat.glsl          战斗系统
│   ├── agent_steer.glsl           转向 + 运动积分（含被挤开时的逃逸方向）
│   ├── swe_flux.glsl              SWE flux 更新
│   ├── swe_velocity.glsl          SWE → 速度场转换
│   ├── goal_bfs.glsl              多阵营 BFS 目标距离场（分块松弛）
│   ├── cell_blocked.glsl          阵营通行性标记
│   ├── build_multimesh.glsl       GPU 端 MultiMesh 缓冲构建
│   ├── laser_compact.glsl         远程激光压缩（供批量绘制）
│   ├── select_inspect.glsl        选中 agent 的 GPU 查询
│   ├── build_overlay.glsl         调试叠加层（热力图/箭头/阻挡格）
│   ├── gi_*.glsl                  GI 各 pass（见 MEMORY.md）
│   ├── std_compact_generic.glsl   通用稀疏 tile 压缩（当前未启用）
│   ├── std_compact_tiles.glsl     tile 坐标压缩（未被引用）
│   └── std_finalize_compact.glsl  indirect dispatch 参数生成（当前未启用）
├── castle_bw.png                  默认城堡模板图
└── project.godot
```

## 核心系统

### 浅水方程（SWE）流体场

将 agent 密度视为水面高度，叠加目标距离场和地形壁作为势能。通过四方向 flux 传播，自然形成从高密度区向低密度区的流动。

- **swe_flux.glsl**：共享内存优化，8×8 工作组 + halo 读取。
- **swe_velocity.glsl**：flux 差分 + 目标梯度混合，密度自适应减速。多目标组（z 维度 dispatch）。

### 稀疏 Tile 调度

基于 UE SparseTileDispatch 移植，将活跃 tile 压缩为紧凑列表，生成 indirect dispatch 参数，让 SWE flux 和 velocity 只在有 agent 活动的区域执行。

当前未启用：`GPUSWEField.sparse_enabled` 为 `false`，也没有派发 compact，SWE flux / velocity 走全图 dispatch。若要启用，注意 `std_finalize_compact.glsl` 写出的 indirect 参数 z = 1，velocity 的 32 个目标组（z 维）只会算第 0 组。

### 空间哈希

4-pass GPU 排序，支持 O(1) 邻居查询：

1. **Clear**：cell_count 清零
2. **Count**：每个 agent 对所属 cell 原子 +1
3. **Prefix Sum**：单个 1024 线程工作组，按 4096 格分块做 exclusive scan（读写经 shared memory 中转以合并访问），计算 cell_start
4. **Scatter**：agent 索引写入 sorted_idx

### 目标场（BFS）

每帧只更新 1 个目标组（32 组轮转），写出 `goal_dist_all` / `gdir_x` / `gdir_y`，下一帧的 velocity 和 combat 使用。BFS 与主模拟在同一次提交里执行。

1. **Init**：有敌方 presence 的格为 0，墙为 `0xFFFFFFFF`，其余为未到达
2. **分块松弛**：每个工作组把 16×16 tile + 1 格 halo 读进 shared memory，在 tile 内迭代到不再变化再写回，每次全局 pass 波前至少推进一个 tile。最多 80 次 pass（覆盖全图对角）；某次 pass 全图无变化后，其余 pass 整组提前退出
3. **Gradient**：写出 float 距离和归一化梯度

### 战斗系统

- 5 种装备：Sword / Spear / Bow / Crossbow / Shield
- 距离场门控：目标场距离大于射程 + 2 格、且 3×3 格内没有敌方 presence 时跳过索敌。因此远程单位不会攻击测地距离很远的敌人（例如隔着墙）
- 阵营 presence bitmap：32-bit 位掩码快速判断敌我
- 攻击时冻结移动；同一格被多名攻击者占用时，其余单位被挤开（displaced），在 steer 中就地扫描 7×7 邻域，向最近的空闲格逃逸

### 远程攻击激光

- combat 在远程单位出手时写 per-agent 的 `laser_line` / `laser_ttl`，GI 的 `gi_laser_emit.glsl` 直接读这两块缓冲
- 屏幕绘制：`laser_compact.glsl` 把可见激光压缩成紧凑列表（最多 8192 条），CPU 回读后由 soft / glow / core 三个 additive 图层各调用一次 `draw_multiline_colors`

### 选中 agent

选中后每帧由 `select_inspect.glsl` 在 GPU 上查询自身状态、最近敌人和所在格的 blocker，只回读 64 字节。所在 group 的阻挡格用 `build_overlay.glsl` 的 mode 2 生成纹理，在选中圈下方整张绘制。

### Agent 转向

- 双线性采样速度场获取目标方向
- 空间哈希 3×3 邻域分离力
- 墙壁方向检测 + 排斥力
- 高密度区随机噪声打破对称
- 拥堵检测 + 切向逃逸
- 地形碰撞轴分离 + 卡墙自动推出
- 阵营感知的硬最小距离投影

### GPU MultiMesh 构建

`build_multimesh.glsl` 把 agent 位置和颜色直接写成 MultiMesh 的缓冲布局（每实例 Transform2D + 颜色，12 个 float）。模拟跑在本地 RenderingDevice 上，与渲染用的主设备不共享资源，所以这块缓冲随主提交异步回读后再用 `MultiMesh.set_buffer` 上传。

### 调试叠加层

- **密度热力图**（mode 0）：红色透明度映射 agent 密度
- **速度场箭头**（mode 3）：HSV 编码方向的彩色箭头，长度随速度缩放
- **目标距离场**（mode 1）：绿色梯度显示 BFS 距离（shader 支持，UI 未暴露）
- **通行性标记**（mode 2）：红色高亮被阵营墙壁阻挡的格子；选中 agent 时自动显示其所在 group 的阻挡格

## 城堡模板

支持通过黑白 PNG 图生成城堡地形。白色像素为墙壁，黑色为通路。图片缩放到 Size 滑块指定的尺寸后，随机放置到地图上。每个阵营获得一座城堡。

UI 中可选择不同 PNG 文件和调整城堡大小（20–120），Reset 后生效。

## 操作说明

| 操作 | 按键/鼠标 |
|------|-----------|
| 暂停/继续 | Space |
| 重置模拟 | R |
| 速度场 | V |
| GI 开关 | L |
| 截图（`user://screenshot_*.png`） | F12 |
| 选择 agent | 左键点击 |
| 取消选择 | 右键 |
| 画刷：无 | 1 |
| 画刷：墙壁 | 2 |
| 画刷：擦除 | 3 |
| 画刷绘制 | 左键拖拽 |
| 画刷擦除 | 右键拖拽 |
| 画刷大小 | 滚轮（画刷模式下） |
| 缩放 | 滚轮（非画刷模式） |
| 平移 | WASD / 中键拖拽 |

右侧面板：agent 数量、音乐音量、城堡模板及大小、密度热力图 / 速度场开关、GI 精度、暂停 / 重置、画刷半径。

## 运行

需要支持 Vulkan 的 GPU 和自编译 Godot 4.6 引擎：

```powershell
& "D:\Godot\godot-source\bin\godot.windows.editor.x86_64.console.exe" --path "D:\MyProject\FluidCrowd"
```

## 导出

```powershell
& "D:\Godot\godot-source\bin\godot.windows.editor.x86_64.console.exe" --path "D:\MyProject\FluidCrowd" --headless --export-release "Windows Desktop" "build/FluidCrowd.exe"
```

输出单文件可执行（PCK 嵌入），约 73 MB。

## 技术要点

- **全 GPU 计算**：所有仿真逻辑在 compute shader 中执行，CPU 仅负责 dispatch 和少量 readback
- **GPU MultiMesh**：agent 渲染数据在 GPU 端构建，随主提交异步回读后上传（本地 RD 与主渲染设备不共享资源）
- **异步回读**：每帧回读都在 `compute_list_end` 之后用 `buffer_get_data_async` 登记，随 `sync()` 一起到达，不额外 flush/stall
- **SoA 数据布局**：pos_x / pos_y / vel_x / vel_y 分离存储，GPU 内存访问合并
- **双缓冲**：位置和速度使用 ping-pong buffer，避免读写竞争
- **共享内存优化**：SWE flux、前缀和、BFS 分块松弛使用 shared memory 减少全局内存访问
- **Push Constant**：小参数通过 push constant 直接传入，无需 uniform buffer
- **PCG 伪随机**：GPU 端确定性随机数，每帧种子 + 线程 ID 组合
