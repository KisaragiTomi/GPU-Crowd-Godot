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

```
crowd_sim.gd               主控：初始化、仿真循环、UI、输入、渲染
├── gpu_swe_field.gd        流体场：SWE flux/velocity、目标场、地形
├── gpu_agents.gd           Agent GPU 管理：buffer、dispatch、MultiMesh
└── sparse_tile_dispatch.gd 稀疏 tile 调度：活跃区域压缩 + indirect dispatch
```

## GPU 管线（每帧）

单次 `compute_list_begin/end` 中按序 dispatch：

```
┌─────────────────────────────────────────────────────┐
│ 0. Sparse Tile Compact                              │
│    density + terrain → 活跃 tile 列表 + indirect args│
├─────────────────────────────────────────────────────┤
│ 1. Spatial Hash Build                               │
│    clear_grid → count_cells → prefix_sum → scatter  │
├─────────────────────────────────────────────────────┤
│ 2. Density                                          │
│    双线性 splatting → density_out + faction_presence │
├─────────────────────────────────────────────────────┤
│ 3. SWE Flux（节流，~33ms 间隔）                       │
│    共享内存加载 H 场 → 四方向 flux 更新（稀疏调度）   │
├─────────────────────────────────────────────────────┤
│ 4. Velocity Field（多目标组）                         │
│    flux 差分 + 目标梯度 → 每 cell 速度向量（稀疏调度）│
├─────────────────────────────────────────────────────┤
│ 5. Combat                                           │
│    距离场门控 → 空间哈希邻居扫描 → 攻击/受伤/死亡      │
├─────────────────────────────────────────────────────┤
│ 6. Steer + Integrate                                │
│    目标速度采样 + 分离力 + 墙壁回避 + 拥堵逃逸         │
│    → 地形碰撞 → 硬距离投影 → 位置更新                  │
├─────────────────────────────────────────────────────┤
│ 7. Build MultiMesh（GPU 端）                         │
│    位置 + 阵营颜色 → Transform2D 直写 MultiMesh      │
├─────────────────────────────────────────────────────┤
│ 8. Build Overlay（可选）                              │
│    density heatmap / velocity arrows / goal dist     │
└─────────────────────────────────────────────────────┘
```

各阶段之间由 `compute_list_add_barrier` 同步。

## 文件结构

```
FluidCrowd/
├── scenes/
│   └── main.tscn                  主场景（CrowdSim + Camera2D）
├── scripts/
│   ├── crowd_sim.gd               主控脚本
│   ├── gpu_agents.gd              Agent GPU 管理
│   ├── gpu_swe_field.gd           SWE 流体场
│   └── sparse_tile_dispatch.gd    稀疏 tile 调度模块
├── shaders/
│   ├── agent_clear_grid.glsl      空间哈希：清零
│   ├── agent_count_cells.glsl     空间哈希：原子计数
│   ├── agent_prefix_sum.glsl      空间哈希：前缀和
│   ├── agent_scatter.glsl         空间哈希：索引散射
│   ├── agent_density.glsl         密度 + 阵营 presence 计算
│   ├── agent_combat.glsl          战斗系统
│   ├── agent_steer.glsl           转向 + 运动积分
│   ├── swe_flux.glsl              SWE flux 更新（稀疏 tile）
│   ├── swe_velocity.glsl          SWE → 速度场转换（稀疏 tile）
│   ├── disp_flow.glsl             目标距离场流向计算
│   ├── goal_bfs.glsl              多阵营 BFS 目标距离场
│   ├── cell_blocked.glsl          阵营通行性标记
│   ├── build_multimesh.glsl       GPU 端 MultiMesh 构建
│   ├── build_overlay.glsl         调试叠加层（热力图/箭头）
│   ├── std_compact_generic.glsl   通用稀疏 tile 压缩
│   ├── std_compact_tiles.glsl     tile 坐标压缩
│   └── std_finalize_compact.glsl  indirect dispatch 参数生成
├── castle_bw.png                  默认城堡模板图
└── project.godot
```

## 核心系统

### 浅水方程（SWE）流体场

将 agent 密度视为水面高度，叠加目标距离场和地形壁作为势能。通过四方向 flux 传播，自然形成从高密度区向低密度区的流动。

- **swe_flux.glsl**：共享内存优化，8×8 工作组 + halo 读取。稀疏 tile 跳过空闲区域。
- **swe_velocity.glsl**：flux 差分 + 目标梯度混合，密度自适应减速。多目标组（z 维度 dispatch）。

### 稀疏 Tile 调度

基于 UE SparseTileDispatch 移植，将活跃 tile 压缩为紧凑列表，生成 indirect dispatch 参数。SWE flux 和 velocity 仅在有 agent 活动的区域执行，大幅降低空闲区域 GPU 开销。

### 空间哈希

4-pass GPU 排序，支持 O(1) 邻居查询：

1. **Clear**：cell_count 清零
2. **Count**：每个 agent 对所属 cell 原子 +1
3. **Prefix Sum**：单工作组内完成，计算 cell_start
4. **Scatter**：agent 索引写入 sorted_idx

### 战斗系统

- 5 种装备：Sword / Spear / Bow / Crossbow / Shield
- 距离场门控：远离敌人时跳过索敌，节省 GPU
- 阵营 presence bitmap：32-bit 位掩码快速判断敌我
- 攻击时冻结移动，被挤占时自动位移到邻近空闲格

### Agent 转向

- 双线性采样速度场获取目标方向
- 空间哈希 3×3 邻域分离力
- 墙壁方向检测 + 排斥力
- 高密度区随机噪声打破对称
- 拥堵检测 + 切向逃逸
- 地形碰撞轴分离 + 卡墙自动推出
- 阵营感知的硬最小距离投影

### GPU MultiMesh 构建

agent 位置和颜色直接在 compute shader 中写入 MultiMesh buffer，避免 CPU readback + 上传开销。

### 调试叠加层

- **密度热力图**（mode 0）：红色透明度映射 agent 密度
- **速度场箭头**（mode 3）：HSV 编码方向的彩色箭头，长度随速度缩放
- **目标距离场**（mode 1）：绿色梯度显示 BFS 距离
- **通行性标记**（mode 2）：红色高亮被阵营墙壁阻挡的格子

## 城堡模板

支持通过黑白 PNG 图生成城堡地形。白色像素为墙壁，黑色为通路。图片缩放到 Size 滑块指定的尺寸后，随机放置到地图上。每个阵营获得一座城堡。

UI 中可选择不同 PNG 文件和调整城堡大小（20–120），Reset 后生效。

## 操作说明

| 操作 | 按键/鼠标 |
|------|-----------|
| 暂停/继续 | Space |
| 重置模拟 | R |
| 密度热力图 | D |
| 速度场 | V |
| 目标距离场 | G |
| 选择 agent | 左键点击 |
| 取消选择 | 右键 |
| 画刷：无 | 1 |
| 画刷：墙壁 | 2 |
| 画刷：目标区 | 3 |
| 画刷：擦除 | 4 |
| 画刷绘制 | 左键拖拽 |
| 画刷擦除 | 右键拖拽 |
| 画刷大小 | 滚轮（画刷模式下） |
| 缩放 | 滚轮（非画刷模式） |
| 平移 | 中键拖拽 |

右侧面板提供滑块控制：agent 数量、移动速度、密度压力、分离力、交战距离、攻击冷却、城堡模板及大小。

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
- **GPU MultiMesh**：agent 渲染数据直接在 GPU 端构建，无 CPU 往返
- **SoA 数据布局**：pos_x / pos_y / vel_x / vel_y 分离存储，GPU 内存访问合并
- **双缓冲**：位置和速度使用 ping-pong buffer，避免读写竞争
- **共享内存优化**：SWE flux shader 使用 shared memory 减少全局内存访问
- **Push Constant**：小参数通过 push constant 直接传入，无需 uniform buffer
- **PCG 伪随机**：GPU 端确定性随机数，每帧种子 + 线程 ID 组合
