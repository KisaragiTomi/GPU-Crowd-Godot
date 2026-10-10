class_name GPUGI
extends RefCounted

## Radiance Cascades GI driven by an agent emission map.
##
## Per GI update:
## 1. 清空 emission 累加缓冲并写入墙体遮挡（一个逐像素 pass）。
## 2. 一次 dispatch 画所有 agent 光斑和远程激光（激光读 GPUAgents 的紧凑列表，每条分段并行），atomicMax 累加。
## 3. 累加结果转成 RGBA16F emission 纹理，同时生成 JFA 种子（R32UI 像素坐标）。
## 4. JFA → R32F 距离场。
## 5. Radiance cascades：从最高级到 0 级逐级派发，每级只覆盖本级像素；所有级紧凑存放在一张 RGBA16F 纹理里。
## 6. 把 cascade 0 解算成 tonemap 后的 GI 图。
## 7. 空间降噪 + 时间稳定（history 为 RGBA16F），同时写 RGBA8 显示纹理供回读。

const DEFAULT_NUM_CASCADES := 6
const DEFAULT_MERGE_FIX := 4
const CASCADE_TEX_W := 2048  # cascade 紧凑纹理的宽度，须与 gi_rc_cube_a / gi_rc_image 的 CASCADE_TEX_W 一致
const LASER_THREADS := 16  # 每条激光的线程数，须与 gi_agent_emit.glsl 一致
const PARAMS_SIZE := 96

var rd: RenderingDevice

var tex_w: int
var tex_h: int
var num_cascades := DEFAULT_NUM_CASCADES
var merge_fix := DEFAULT_MERGE_FIX
var jfa_passes: int
# 每级 cascade 的像素数与线性偏移（与 shader 的 GetCascadeSize / GetCascadeLinearOffset 同一公式）
var cascade_counts := PackedInt32Array()
var cascade_offsets := PackedInt32Array()
var cascade_tex_h: int

var params_buf: RID  # 每次 GI 更新写一次，布局见 _update_params
var buf_emit_acc: RID  # 每像素 uvec4：emission rgba 的 fp32 位模式，供 atomicMax
var jfa_params_bufs: Array[RID] = []
var cascade_params_bufs: Array[RID] = []

var tex_emit: RID  # RGBA16F
var tex_jfa: Array[RID] = []  # R32UI ping-pong：最近种子的像素坐标 x | y << 16，0 = 无
var tex_distance: RID  # R32F
var tex_cascades: RID  # RGBA16F，CASCADE_TEX_W × cascade_tex_h
var tex_output: RID  # RGBA16F，tonemap 后的 GI
var tex_history: Array[RID] = []  # RGBA16F ping-pong
var tex_display: RID  # RGBA8，回读后显示

var shd_wall_mask: RID
var shd_emit: RID
var shd_seed: RID
var shd_jfa: RID
var shd_distance: RID
var shd_cube_a: RID
var shd_image: RID
var shd_stabilize: RID
var shd_resample: RID

var pip_wall_mask: RID
var pip_emit: RID
var pip_seed: RID
var pip_jfa: RID
var pip_distance: RID
var pip_cube_a: RID
var pip_image: RID
var pip_stabilize: RID
var pip_resample: RID

var buf_pos_x: RID
var buf_pos_y: RID
var buf_agent_info: RID
var buf_fac_colors: RID
var buf_hit_flash: RID
var buf_laser_compact: RID
var laser_cap := 0
var buf_terrain: RID
var terrain_grid_w: int
var terrain_grid_h: int
var terrain_cell_size: float
var laser_width := 0.65
var laser_emit_strength := 3.2
var laser_life := 0.12
var interval_scale := 1.0
var temporal_blend := 0.82
var spatial_filter_strength := 0.65
var temporal_valid := false
var _last_region_origin := Vector2.ZERO
var _last_region_size := Vector2.ZERO

# Debug: 0 = emission map, 1 = JFA seeds, 2 = distance field,
# 3 = cascade storage (all cascades packed), 4 = final GI.
var debug_stage: int = 4

# uniform set 全部预建 / 缓存，不再每次 GI 更新新建再释放
var _us_seed: RID
var _us_jfa: Array[RID] = []  # 第 k 遍：读 tex_jfa[k % 2]，写 tex_jfa[(k + 1) % 2]
var _us_distance: RID
var _us_cascade: Array[RID] = []
var _us_image: RID
var _us_stabilize: Array[RID] = []  # [h]：读 tex_history[h]，写 tex_history[1 - h]
var _us_resample: RID
var _us_wall: RID
var _us_wall_terrain: RID  # _us_wall 绑定的地形缓冲
var _us_emit := {}  # [pos_x, pos_y, info, colors, hit_flash, laser] -> uniform set（位置是双缓冲，通常两份）
var _hist := 0  # 当前 history 在 tex_history 里的下标
var _emit_agents := 0  # 本次 emission dispatch 的 agent 线程数
var _emit_lasers := 0  # 本次 emission dispatch 的激光条数
var _out_raw := PackedByteArray()  # _finish_compute 登记、随 sync() 到达的显示纹理数据


func _init(rendering_device: RenderingDevice, width: int, height: int,
		p_num_cascades: int = DEFAULT_NUM_CASCADES,
		p_merge_fix: int = DEFAULT_MERGE_FIX) -> void:
	rd = rendering_device
	tex_w = width
	tex_h = height
	num_cascades = clampi(p_num_cascades, 1, 6)
	merge_fix = clampi(p_merge_fix, 0, 5)
	jfa_passes = ceili(log(float(maxi(tex_w, tex_h))) / log(2.0))
	_compute_cascade_layout()

	params_buf = rd.storage_buffer_create(PARAMS_SIZE, _zero_bytes(PARAMS_SIZE / 4))
	buf_emit_acc = rd.storage_buffer_create(tex_w * tex_h * 16)
	_create_static_param_buffers()

	var rgba16f := RenderingDevice.DATA_FORMAT_R16G16B16A16_SFLOAT
	tex_emit = _create_tex(tex_w, tex_h, rgba16f)
	for _i in range(2):
		tex_jfa.append(_create_tex(tex_w, tex_h, RenderingDevice.DATA_FORMAT_R32_UINT))
		tex_history.append(_create_tex(tex_w, tex_h, rgba16f))
	tex_distance = _create_tex(tex_w, tex_h, RenderingDevice.DATA_FORMAT_R32_SFLOAT)
	tex_cascades = _create_tex(CASCADE_TEX_W, cascade_tex_h, rgba16f)
	tex_output = _create_tex(tex_w, tex_h, rgba16f)
	tex_display = _create_tex(tex_w, tex_h, RenderingDevice.DATA_FORMAT_R8G8B8A8_UNORM)
	# history 第一次以权重 0 参与混合，0 × 未初始化的 NaN 仍是 NaN，先清零
	for tex in tex_history + [tex_display]:
		rd.texture_clear(tex, Color(0, 0, 0, 0), 0, 1, 0, 1)

	_load_shaders()
	_build_static_uniform_sets()


func set_agent_buffers(px: RID, py: RID, info: RID, colors: RID, hit_flash: RID) -> void:
	buf_pos_x = px
	buf_pos_y = py
	buf_agent_info = info
	buf_fac_colors = colors
	buf_hit_flash = hit_flash


## compact：GPUAgents.buf_laser_compact（主提交里 laser_compact.glsl 写好的紧凑激光列表），cap：它的容量 LASER_CAP
func set_laser_buffer(compact: RID, cap: int) -> void:
	buf_laser_compact = compact
	laser_cap = cap


func set_terrain_buffer(terrain: RID, grid_w: int, grid_h: int, cell_size: float) -> void:
	buf_terrain = terrain
	terrain_grid_w = grid_w
	terrain_grid_h = grid_h
	terrain_cell_size = cell_size


## laser_count：本帧压缩出的激光条数（crowd_sim._laser_count，可能超过 cap，超出的不发光）
func compute_gi(agent_count: int, region_origin: Vector2, region_size: Vector2,
		splat_radius: float, emit_strength: float, laser_count: int = 0) -> void:
	if not _has_agent_buffers():
		return

	var region_stable := temporal_valid and _is_same_region(region_origin, region_size)
	_update_params(agent_count, region_origin, region_size, splat_radius, emit_strength, laser_count,
			temporal_blend if region_stable else 0.0)
	_last_region_origin = region_origin
	_last_region_size = region_size

	var cl := rd.compute_list_begin()

	_dispatch_clear_walls(cl)
	rd.compute_list_add_barrier(cl)

	_dispatch_emit(cl)
	rd.compute_list_add_barrier(cl)

	_dispatch_seed(cl)
	rd.compute_list_add_barrier(cl)

	if debug_stage == 0 or debug_stage == 1:
		_dispatch_resample(cl)
		_finish_compute(cl)
		return

	_dispatch_jfa(cl)

	_dispatch_distance(cl)
	rd.compute_list_add_barrier(cl)

	if debug_stage == 2:
		_dispatch_resample(cl)
		_finish_compute(cl)
		return

	_dispatch_radiance_cascades(cl)

	if debug_stage == 3:
		_dispatch_resample(cl)
		_finish_compute(cl)
		return

	_dispatch_image(cl)
	rd.compute_list_add_barrier(cl)
	_dispatch_stabilize(cl)
	_finish_compute(cl)
	_hist = 1 - _hist
	temporal_valid = true


func get_output_image() -> Image:
	var raw := _out_raw
	_out_raw = PackedByteArray()
	# 优先用 _finish_compute 登记、随 sync() 到达的数据；缺失时退回同步回读
	if raw.size() != tex_w * tex_h * 4: raw = rd.texture_get_data(tex_display, 0)
	return Image.create_from_data(tex_w, tex_h, false, Image.FORMAT_RGBA8, raw)


func cleanup() -> void:
	if rd == null:
		return

	# uniform set 先于其引用的纹理 / 缓冲 / shader 释放（依赖若已在别处释放，Godot 已连带释放了 set）
	var sets: Array[RID] = [_us_wall, _us_seed, _us_distance, _us_image, _us_resample]
	sets.append_array(_us_jfa)
	sets.append_array(_us_cascade)
	sets.append_array(_us_stabilize)
	for us in _us_emit.values():
		sets.append(us)
	for us in sets:
		_free_uniform_set(us)
	_us_emit.clear()

	var free_rid := func(rid: RID) -> void:
		if rid.is_valid():
			rd.free_rid(rid)

	for rid in [
		tex_emit, tex_distance, tex_cascades, tex_output, tex_display,
		params_buf, buf_emit_acc,
		pip_wall_mask, pip_emit, pip_seed, pip_jfa, pip_distance, pip_cube_a, pip_image, pip_stabilize, pip_resample,
		shd_wall_mask, shd_emit, shd_seed, shd_jfa, shd_distance, shd_cube_a, shd_image, shd_stabilize, shd_resample,
	]:
		free_rid.call(rid)

	for arr in [tex_jfa, tex_history, jfa_params_bufs, cascade_params_bufs]:
		for rid in arr:
			free_rid.call(rid)

	rd = null


func _has_agent_buffers() -> bool:
	return buf_pos_x.is_valid() and buf_pos_y.is_valid() and buf_agent_info.is_valid() and buf_fac_colors.is_valid() and buf_hit_flash.is_valid()


func _dispatch_clear_walls(cl: int) -> void:
	rd.compute_list_bind_compute_pipeline(cl, pip_wall_mask)
	rd.compute_list_bind_uniform_set(cl, _wall_uniform_set(), 0)
	rd.compute_list_dispatch(cl, ceili(float(tex_w) / 16.0), ceili(float(tex_h) / 16.0), 1)


func _dispatch_emit(cl: int) -> void:
	var threads := _emit_agents + _emit_lasers * LASER_THREADS
	if threads <= 0:
		return
	rd.compute_list_bind_compute_pipeline(cl, pip_emit)
	rd.compute_list_bind_uniform_set(cl, _emit_uniform_set(), 0)
	rd.compute_list_dispatch(cl, ceili(float(threads) / 64.0), 1, 1)


func _dispatch_seed(cl: int) -> void:
	rd.compute_list_bind_compute_pipeline(cl, pip_seed)
	rd.compute_list_bind_uniform_set(cl, _us_seed, 0)
	rd.compute_list_dispatch(cl, ceili(float(tex_w) / 16.0), ceili(float(tex_h) / 16.0), 1)


func _dispatch_jfa(cl: int) -> void:
	for k in range(jfa_passes):
		rd.compute_list_bind_compute_pipeline(cl, pip_jfa)
		rd.compute_list_bind_uniform_set(cl, _us_jfa[k], 0)
		rd.compute_list_dispatch(cl, ceili(float(tex_w) / 16.0), ceili(float(tex_h) / 16.0), 1)
		rd.compute_list_add_barrier(cl)


func _dispatch_distance(cl: int) -> void:
	rd.compute_list_bind_compute_pipeline(cl, pip_distance)
	rd.compute_list_bind_uniform_set(cl, _us_distance, 0)
	rd.compute_list_dispatch(cl, ceili(float(tex_w) / 16.0), ceili(float(tex_h) / 16.0), 1)


func _dispatch_cascade(cl: int, c: int) -> void:
	# 一个线程一个本级像素；组数超过 65535 时拆成二维（shader 按 gl_NumWorkGroups 还原线性下标）
	var groups := ceili(float(cascade_counts[c]) / 64.0)
	var gx := mini(groups, 65535)
	rd.compute_list_bind_compute_pipeline(cl, pip_cube_a)
	rd.compute_list_bind_uniform_set(cl, _us_cascade[c], 0)
	rd.compute_list_dispatch(cl, gx, ceili(float(groups) / float(gx)), 1)


func _dispatch_radiance_cascades(cl: int) -> void:
	# 从最高级到 0 级：第 c 级读第 c + 1 级，级间 barrier 保证读到的是本次更新刚写完的数据
	for c in range(num_cascades - 1, -1, -1):
		_dispatch_cascade(cl, c)
		rd.compute_list_add_barrier(cl)


func _dispatch_image(cl: int) -> void:
	rd.compute_list_bind_compute_pipeline(cl, pip_image)
	rd.compute_list_bind_uniform_set(cl, _us_image, 0)
	rd.compute_list_dispatch(cl, ceili(float(tex_w) / 16.0), ceili(float(tex_h) / 16.0), 1)


func _dispatch_stabilize(cl: int) -> void:
	rd.compute_list_bind_compute_pipeline(cl, pip_stabilize)
	rd.compute_list_bind_uniform_set(cl, _us_stabilize[_hist], 0)
	rd.compute_list_dispatch(cl, ceili(float(tex_w) / 16.0), ceili(float(tex_h) / 16.0), 1)


# debug_stage 0..3：把对应的中间结果重采样到显示纹理
func _dispatch_resample(cl: int) -> void:
	rd.compute_list_bind_compute_pipeline(cl, pip_resample)
	rd.compute_list_bind_uniform_set(cl, _us_resample, 0)
	rd.compute_list_dispatch(cl, ceili(float(tex_w) / 16.0), ceili(float(tex_h) / 16.0), 1)


func _finish_compute(cl: int) -> void:
	rd.compute_list_end()
	# 登记显示纹理的异步回读，数据随 sync() 到达
	_out_raw = PackedByteArray()
	rd.texture_get_data_async(tex_display, 0, _on_output_data)
	rd.submit()
	rd.sync()


func _on_output_data(data: PackedByteArray) -> void:
	_out_raw = data


func _is_same_region(region_origin: Vector2, region_size: Vector2) -> bool:
	if _last_region_size == Vector2.ZERO:
		return false
	return (
		_last_region_origin.distance_to(region_origin) < 0.25
		and _last_region_size.distance_to(region_size) < 0.25
	)


# 与 gi_rc_cube_a.glsl 的 GetC0Size / GetCascadeSize / GetCascadeLinearOffset 相同：
# c0 探针数 (512, 512·h/w)、方向数 4；第 i 级探针数每轴 >> i（至少 1），方向数 × 4^i
func _compute_cascade_layout() -> void:
	var c0 := Vector2i(512, (512 * tex_h) / tex_w)
	cascade_counts.resize(num_cascades)
	cascade_offsets.resize(num_cascades)
	var offset := 0
	for i in range(num_cascades):
		cascade_offsets[i] = offset
		cascade_counts[i] = maxi(1, c0.x >> i) * maxi(1, c0.y >> i) * (4 << (2 * i))
		offset += cascade_counts[i]
	cascade_tex_h = ceili(float(offset) / float(CASCADE_TEX_W))


func _create_static_param_buffers() -> void:
	jfa_params_bufs.clear()
	for k in range(jfa_passes):
		# 第 k 遍的跳距：2^(jfa_passes - 1) … 1
		var jfa_bytes := PackedFloat32Array([pow(2.0, jfa_passes - 1 - k)]).to_byte_array()
		jfa_bytes.resize(16)
		jfa_params_bufs.append(rd.storage_buffer_create(jfa_bytes.size(), jfa_bytes))

	cascade_params_bufs.clear()
	for c in range(num_cascades):
		var cascade_bytes := PackedInt32Array([c, cascade_counts[c], 0, 0]).to_byte_array()
		cascade_params_bufs.append(rd.storage_buffer_create(cascade_bytes.size(), cascade_bytes))


# 布局（字节偏移）与各 shader 的 ParamsBuffer / GIParamsBuffer 一致（各取所需的前缀）：
# 0 agent_count, 4 tex_w, 8 tex_h, 12..24 region xywh, 28 splat_radius, 32 emit_strength,
# 36 terrain_w, 40 terrain_h, 44 terrain_cell_size, 48 wall_alpha, 52 laser_width,
# 56 laser_emit_strength, 60 laser_life, 64 laser_count, 68 laser_cap, 72 interval_scale,
# 76 history_weight, 80 spatial_strength, 84 num_cascades, 88 merge_fix, 92 debug_stage
func _update_params(agent_count: int, region_origin: Vector2, region_size: Vector2,
		splat_radius: float, emit_strength: float, laser_count: int, history_weight: float) -> void:
	_emit_agents = maxi(agent_count, 0)
	var lasers_ok := buf_laser_compact.is_valid() and laser_cap > 0 and laser_life > 0.0
	_emit_lasers = clampi(laser_count, 0, laser_cap) if lasers_ok else 0
	var has_terrain := buf_terrain.is_valid()
	var bytes := PackedByteArray()
	bytes.resize(PARAMS_SIZE)
	bytes.encode_s32(0, _emit_agents)
	bytes.encode_float(4, float(tex_w))
	bytes.encode_float(8, float(tex_h))
	bytes.encode_float(12, region_origin.x)
	bytes.encode_float(16, region_origin.y)
	bytes.encode_float(20, region_size.x)
	bytes.encode_float(24, region_size.y)
	bytes.encode_float(28, splat_radius)
	bytes.encode_float(32, emit_strength)
	bytes.encode_s32(36, terrain_grid_w if has_terrain else 0)
	bytes.encode_s32(40, terrain_grid_h if has_terrain else 0)
	bytes.encode_float(44, terrain_cell_size)
	bytes.encode_float(48, 1.0)
	bytes.encode_float(52, laser_width)
	bytes.encode_float(56, laser_emit_strength)
	bytes.encode_float(60, laser_life)
	bytes.encode_s32(64, _emit_lasers)
	bytes.encode_s32(68, laser_cap)
	bytes.encode_float(72, clampf(interval_scale, 0.125, 8.0))
	bytes.encode_float(76, clampf(history_weight, 0.0, 0.98))
	bytes.encode_float(80, clampf(spatial_filter_strength, 0.0, 1.0))
	bytes.encode_s32(84, num_cascades)
	bytes.encode_s32(88, merge_fix)
	bytes.encode_s32(92, debug_stage)
	rd.buffer_update(params_buf, 0, bytes.size(), bytes)


func _build_static_uniform_sets() -> void:
	_us_seed = _uniform_set(shd_seed, [
		_buf_uniform(0, buf_emit_acc),
		_img_uniform(1, tex_emit),
		_img_uniform(2, tex_jfa[0]),
	])
	_us_jfa.clear()
	for k in range(jfa_passes):
		_us_jfa.append(_uniform_set(shd_jfa, [
			_img_uniform(1, tex_jfa[(k + 1) % 2]),
			_img_uniform(2, tex_jfa[k % 2]),
			_buf_uniform(4, jfa_params_bufs[k]),
		]))
	_us_distance = _uniform_set(shd_distance, [
		_img_uniform(1, tex_distance),
		_img_uniform(2, tex_jfa[jfa_passes % 2]),
	])
	_us_cascade.clear()
	for c in range(num_cascades):
		_us_cascade.append(_uniform_set(shd_cube_a, [
			_buf_uniform(0, params_buf),
			_buf_uniform(1, cascade_params_bufs[c]),
			_img_uniform(2, tex_emit),
			_img_uniform(3, tex_distance),
			_img_uniform(10, tex_cascades),
		]))
	_us_image = _uniform_set(shd_image, [
		_buf_uniform(0, params_buf),
		_img_uniform(1, tex_output),
		_img_uniform(10, tex_cascades),
	])
	_us_stabilize.clear()
	for h in range(2):
		_us_stabilize.append(_uniform_set(shd_stabilize, [
			_img_uniform(0, tex_output),
			_img_uniform(1, tex_history[h]),
			_img_uniform(2, tex_history[1 - h]),
			_buf_uniform(3, params_buf),
			_img_uniform(4, tex_display),
		]))
	_us_resample = _uniform_set(shd_resample, [
		_buf_uniform(0, params_buf),
		_img_uniform(1, tex_display),
		_img_uniform(2, tex_emit),
		_img_uniform(3, tex_jfa[0]),
		_img_uniform(4, tex_distance),
		_img_uniform(5, tex_cascades),
	])


func _wall_uniform_set() -> RID:
	# 没有地形缓冲时绑一个占位缓冲，参数里 terrain_w = 0 让 shader 跳过墙体
	var terrain := buf_terrain if buf_terrain.is_valid() else params_buf
	if _us_wall_terrain == terrain and rd.uniform_set_is_valid(_us_wall): return _us_wall
	_free_uniform_set(_us_wall)
	_us_wall = _uniform_set(shd_wall_mask, [
		_buf_uniform(0, terrain),
		_buf_uniform(1, buf_emit_acc),
		_buf_uniform(2, params_buf),
	])
	_us_wall_terrain = terrain
	return _us_wall


func _emit_uniform_set() -> RID:
	var laser := buf_laser_compact if buf_laser_compact.is_valid() else params_buf
	var key := [buf_pos_x, buf_pos_y, buf_agent_info, buf_fac_colors, buf_hit_flash, laser]
	var us: RID = _us_emit.get(key, RID())
	if rd.uniform_set_is_valid(us): return us
	# 缓冲换了，或被释放（Godot 会连带释放引用它的 set）：丢掉失效项，已有两份时全部重建
	for k in _us_emit.keys():
		if not rd.uniform_set_is_valid(_us_emit[k]): _us_emit.erase(k)
	if _us_emit.size() >= 2:
		for old in _us_emit.values():
			_free_uniform_set(old)
		_us_emit.clear()
	us = _uniform_set(shd_emit, [
		_buf_uniform(0, buf_pos_x),
		_buf_uniform(1, buf_pos_y),
		_buf_uniform(2, buf_agent_info),
		_buf_uniform(3, buf_fac_colors),
		_buf_uniform(4, buf_emit_acc),
		_buf_uniform(5, params_buf),
		_buf_uniform(6, buf_hit_flash),
		_buf_uniform(7, laser),
	])
	_us_emit[key] = us
	return us


func _uniform_set(shader: RID, uniforms: Array) -> RID:
	return rd.uniform_set_create(uniforms, shader, 0)


func _free_uniform_set(us: RID) -> void:
	if us.is_valid() and rd.uniform_set_is_valid(us):
		rd.free_rid(us)


func _create_tex(w: int, h: int, format: RenderingDevice.DataFormat) -> RID:
	var fmt := RDTextureFormat.new()
	fmt.width = w
	fmt.height = h
	fmt.format = format
	fmt.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT
		| RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT
		| RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	)
	return rd.texture_create(fmt, RDTextureView.new())


func _zero_bytes(count: int) -> PackedByteArray:
	var bytes := PackedByteArray()
	bytes.resize(count * 4)
	bytes.fill(0)
	return bytes


func _load_shaders() -> void:
	shd_wall_mask = _load_shader("res://shaders/gi_wall_mask.glsl")
	shd_emit = _load_shader("res://shaders/gi_agent_emit.glsl")
	shd_seed = _load_shader("res://shaders/gi_seed.glsl")
	shd_jfa = _load_shader("res://shaders/gi_jump_flood_algorithm.glsl")
	shd_distance = _load_shader("res://shaders/gi_distance.glsl")
	shd_cube_a = _load_shader("res://shaders/gi_rc_cube_a.glsl")
	shd_image = _load_shader("res://shaders/gi_rc_image.glsl")
	shd_stabilize = _load_shader("res://shaders/gi_stabilize.glsl")
	shd_resample = _load_shader("res://shaders/gi_resample_image.glsl")

	pip_wall_mask = rd.compute_pipeline_create(shd_wall_mask)
	pip_emit = rd.compute_pipeline_create(shd_emit)
	pip_seed = rd.compute_pipeline_create(shd_seed)
	pip_jfa = rd.compute_pipeline_create(shd_jfa)
	pip_distance = rd.compute_pipeline_create(shd_distance)
	pip_cube_a = rd.compute_pipeline_create(shd_cube_a)
	pip_image = rd.compute_pipeline_create(shd_image)
	pip_stabilize = rd.compute_pipeline_create(shd_stabilize)
	pip_resample = rd.compute_pipeline_create(shd_resample)


func _load_shader(path: String) -> RID:
	var file := load(path) as RDShaderFile
	if file == null:
		push_error("[GPUGI] Could not load shader: " + path)
		return RID()

	var spirv := file.get_spirv()
	var err := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not err.is_empty():
		push_error("[GPUGI] Shader compile error in %s:\n%s" % [path, err])
	return rd.shader_create_from_spirv(spirv)


func _img_uniform(binding: int, tex: RID) -> RDUniform:
	var u := RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u.binding = binding
	u.add_id(tex)
	return u


func _buf_uniform(binding: int, buf: RID) -> RDUniform:
	var u := RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u.binding = binding
	u.add_id(buf)
	return u
