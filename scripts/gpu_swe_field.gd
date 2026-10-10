class_name GPUSWEField
extends RefCounted

var gw: int
var gh: int
var cs: float
var cell_count: int
var max_groups: int

# CPU-side arrays（目标场 goal_dist / goal_dist_all / gdir 只在 GPU 上：开局由 build_all_goal_fields 生成）
var terrain: PackedFloat32Array
const GROUND_COST_MAX := 4.0
var ground_cost: PackedFloat32Array         # 1 = normal; integer 2..4 = slower ground
var density: PackedFloat32Array             # cell_count (readback)
var out_vx: PackedFloat32Array              # cell_count (readback/display)
var out_vy: PackedFloat32Array              # cell_count (readback/display)

# Tuning
var gravity := 25.0
var damping := 0.90
var density_scale := 2.0
var goal_scale := 1.0
var wall_scale := 500.0
var max_flux := 50.0
var agent_speed := 70.0
var density_slowdown := 0.3

# GPU handles
var rd: RenderingDevice

var buf_density: RID
var buf_goal_dist: RID      # group 0 weighted movement cost (SWE/debug)
var buf_goal_dist_all: RID  # unweighted cell distances for combat range
var buf_terrain: RID
var buf_ground_cost: RID
var buf_fr: RID
var buf_fl: RID
var buf_fd: RID
var buf_fu: RID
var buf_gdir_x: RID
var buf_gdir_y: RID
var buf_out_vx: RID
var buf_out_vy: RID
var buf_bfs_dist: RID
var buf_bfs_cost: RID  # weighted travel distance; bfs_dist stays unweighted for combat range
var buf_bfs_flags: RID  # BFS 全局变化标志（3 槽轮换），用于 early-out；第 4 槽为开局目标场的最大可达距离
var buf_dir_lut: RID    # 开局目标场的梯度方向表（goal_bfs.glsl binding 9），见 _ensure_dir_lut
var uset_bfs: RID
var _dir_lut_cap := 0
var _bfs_presence: RID       # setup_bfs 的参数：方向表扩容时重建 uset_bfs
var _bfs_fac_to_group: RID
var _bfs_flags_raw := PackedByteArray()

var shader_flux: RID
var shader_vel: RID
var shader_bfs: RID
var pipeline_flux: RID
var pipeline_vel: RID
var pipeline_bfs: RID
var uset_flux: RID
var uset_vel: RID

var groups_x: int
var groups_y: int

# Sparse tile dispatch (decoupled module)
var sparse: SparseTileDispatch
var sparse_enabled := false


func _init(rendering_device: RenderingDevice, width: int, height: int,
		   cell_size: float, p_max_groups: int = 1) -> void:
	rd = rendering_device
	gw = width
	gh = height
	cs = cell_size
	cell_count = gw * gh
	max_groups = maxi(p_max_groups, 1)

	terrain       = PackedFloat32Array(); terrain.resize(cell_count)
	ground_cost   = PackedFloat32Array(); ground_cost.resize(cell_count); ground_cost.fill(1.0)
	density       = PackedFloat32Array(); density.resize(cell_count)
	out_vx        = PackedFloat32Array(); out_vx.resize(cell_count)
	out_vy        = PackedFloat32Array(); out_vy.resize(cell_count)

	groups_x = ceili(float(gw) / 8.0)
	groups_y = ceili(float(gh) / 8.0)

	_init_gpu()


func idx(x: int, y: int) -> int:
	return y * gw + x


# ── Environment setup (CPU) ─────────────────────────────────────────────

func add_wall(x0: int, y0: int, x1: int, y1: int) -> void:
	for yy in range(maxi(y0, 0), mini(y1, gh)):
		for xx in range(maxi(x0, 0), mini(x1, gw)):
			terrain[idx(xx, yy)] = 1.0


## 开局目标场（GPU）：同时计算格数距离（战斗）和地面加权距离（移动方向）。
## 普通地面保留原 BFS 结果。调用前墙壁和地面代价须已上传（upload_static_data）。
func build_all_goal_fields(group_goals: Array) -> void:
	if group_goals.is_empty():
		return
	var gx := ceili(float(gw) / BFS_TILE)
	var gy := ceili(float(gh) / BFS_TILE)
	var p := PackedByteArray(); p.resize(32)
	p.encode_s32(0, gw)
	p.encode_s32(4, gh)
	p.encode_s32(28, max_groups)

	# INIT（墙 / 未到达），再由 CPU 写种子 0：越界或是墙的跳过（同原 CPU 版）
	p.encode_s32(8, 3)
	var cl := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, pipeline_bfs)
	rd.compute_list_bind_uniform_set(cl, uset_bfs, 0)
	rd.compute_list_set_push_constant(cl, p, 32)
	rd.compute_list_dispatch(cl, gx, gy, 1)
	rd.compute_list_end()
	var zero4 := PackedByteArray(); zero4.resize(4)
	for g in group_goals[0]:
		var gv := Vector2i(int(g.x), int(g.y))
		if gv.x >= 0 and gv.x < gw and gv.y >= 0 and gv.y < gh and terrain[idx(gv.x, gv.y)] < 0.5:
			rd.buffer_update(buf_bfs_dist, idx(gv.x, gv.y) * 4, 4, zero4)
			rd.buffer_update(buf_bfs_cost, idx(gv.x, gv.y) * 4, 4, zero4)

	var relaxed := 0
	var flags := PackedByteArray()
	while true:
		cl = rd.compute_list_begin()
		rd.compute_list_bind_compute_pipeline(cl, pipeline_bfs)
		rd.compute_list_bind_uniform_set(cl, uset_bfs, 0)
		# RELAX：pass 序号接着上一批编号，early-out 标志槽照常轮换
		p.encode_s32(8, 1)
		for i in range(BFS_RELAX_PASSES):
			if i > 0:
				rd.compute_list_add_barrier(cl)
			p.encode_s32(20, relaxed + i)
			rd.compute_list_set_push_constant(cl, p, 32)
			rd.compute_list_dispatch(cl, gx, gy, 1)
		relaxed += BFS_RELAX_PASSES
		# 最大可达距离 → 写出全部组
		p.encode_s32(24, _dir_lut_cap)
		for mode in [4, 5]:
			rd.compute_list_add_barrier(cl)
			p.encode_s32(8, mode)
			rd.compute_list_set_push_constant(cl, p, 32)
			rd.compute_list_dispatch(cl, gx, gy, 1)
		rd.compute_list_end()
		_bfs_flags_raw = PackedByteArray()
		rd.buffer_get_data_async(buf_bfs_flags, _on_bfs_flags, 0, 16)
		rd.submit()
		rd.sync()
		flags = _bfs_flags_raw
		if flags.size() < 16:
			flags = rd.buffer_get_data(buf_bfs_flags, 0, 16)  # 数据缺失（理论上不会）
		# 最后一个 pass 全图无变化即已收敛；否则（极端地形）清掉最大距离，接着松弛再写一次
		if flags.decode_u32(((relaxed - 1) % 3) * 4) == 0:
			break
		rd.buffer_clear(buf_bfs_flags, 12, 4)

	# |ddx|、|ddy| 最大为填充值 = 最大可达距离 + 1；方向表不够大（极端地形）就扩容后重写
	var need := flags.decode_u32(12) + 2
	if need > _dir_lut_cap:
		_ensure_dir_lut(need)
		p.encode_s32(8, 5)
		p.encode_s32(24, _dir_lut_cap)
		cl = rd.compute_list_begin()
		rd.compute_list_bind_compute_pipeline(cl, pipeline_bfs)
		rd.compute_list_bind_uniform_set(cl, uset_bfs, 0)
		rd.compute_list_set_push_constant(cl, p, 32)
		rd.compute_list_dispatch(cl, gx, gy, 1)
		rd.compute_list_end()
		rd.submit()
		rd.sync()


func _on_bfs_flags(data: PackedByteArray) -> void:
	_bfs_flags_raw = data


const DIR_LUT_MIN_CAP := 1024

## 开局目标场的梯度方向表，覆盖 v ∈ [0, cap)：每个 v 存 (v, s) 和 (v, v + s)（s = 0..2）两组，
## 每组是两个分量 a / sqrt(a² + b²)、b / sqrt(a² + b²)，算式同原 CPU 版（GDScript 双精度，存成 float32）
func _ensure_dir_lut(need: int) -> void:
	if need <= _dir_lut_cap:
		return
	var cap := maxi(_dir_lut_cap, DIR_LUT_MIN_CAP)
	while cap < need:
		cap *= 2
	var lut := PackedFloat32Array(); lut.resize(cap * 12)
	for v in range(cap):
		for s in range(3):
			var k := v * 12 + s * 2
			var la := sqrt(float(v * v + s * s))
			if la > 1e-6:
				lut[k] = v / la
				lut[k + 1] = s / la
			var w := v + s
			var lb := sqrt(float(v * v + w * w))
			if lb > 1e-6:
				lut[k + 6] = v / lb
				lut[k + 7] = w / lb
	# uniform set 引用旧表，要先于旧表释放
	if uset_bfs.is_valid():
		rd.free_rid(uset_bfs)
	if buf_dir_lut.is_valid():
		rd.free_rid(buf_dir_lut)
	var b := lut.to_byte_array()
	buf_dir_lut = rd.storage_buffer_create(b.size(), b)
	_dir_lut_cap = cap
	if _bfs_presence.is_valid():
		setup_bfs(_bfs_presence, _bfs_fac_to_group)


# ── GPU init ─────────────────────────────────────────────────────────────

func _init_gpu() -> void:
	var n := cell_count * 4
	var ng := cell_count * max_groups * 4
	var zb := _zero_bytes(cell_count)
	var zgb := _zero_bytes(cell_count * max_groups)

	buf_density       = rd.storage_buffer_create(n, zb)
	buf_goal_dist     = rd.storage_buffer_create(n, zb)
	buf_goal_dist_all = rd.storage_buffer_create(ng, zgb)
	buf_terrain       = rd.storage_buffer_create(n, zb)
	buf_ground_cost   = rd.storage_buffer_create(n, ground_cost.to_byte_array())
	buf_fr            = rd.storage_buffer_create(n, zb)
	buf_fl            = rd.storage_buffer_create(n, zb)
	buf_fd            = rd.storage_buffer_create(n, zb)
	buf_fu            = rd.storage_buffer_create(n, zb)
	buf_gdir_x        = rd.storage_buffer_create(ng, zgb)
	buf_gdir_y        = rd.storage_buffer_create(ng, zgb)
	buf_out_vx        = rd.storage_buffer_create(ng, zgb)
	buf_out_vy        = rd.storage_buffer_create(ng, zgb)

	var flux_spirv := (load("res://shaders/swe_flux.glsl") as RDShaderFile).get_spirv()
	shader_flux   = rd.shader_create_from_spirv(flux_spirv)
	pipeline_flux = rd.compute_pipeline_create(shader_flux)

	var vel_spirv := (load("res://shaders/swe_velocity.glsl") as RDShaderFile).get_spirv()
	shader_vel   = rd.shader_create_from_spirv(vel_spirv)
	pipeline_vel = rd.compute_pipeline_create(shader_vel)

	var bfs_spirv := (load("res://shaders/goal_bfs.glsl") as RDShaderFile).get_spirv()
	shader_bfs   = rd.shader_create_from_spirv(bfs_spirv)
	pipeline_bfs = rd.compute_pipeline_create(shader_bfs)

	var zb_uint := PackedByteArray()
	zb_uint.resize(cell_count * 4)
	zb_uint.fill(0xFF)
	buf_bfs_dist = rd.storage_buffer_create(cell_count * 4, zb_uint)
	buf_bfs_cost = rd.storage_buffer_create(cell_count * 4, zb_uint)
	var z16 := PackedByteArray(); z16.resize(16)
	buf_bfs_flags = rd.storage_buffer_create(16, z16)
	_ensure_dir_lut(DIR_LUT_MIN_CAP)

	# Sparse tile dispatch (decoupled module)
	sparse = SparseTileDispatch.new(rd, gw, gh, 8, 1, 0.001)
	sparse.set_activity_buffers(buf_density, buf_terrain)

	_build_uniform_sets()


func _build_uniform_sets() -> void:
	uset_flux = rd.uniform_set_create([
		_ubuf(0, buf_density), _ubuf(1, buf_goal_dist), _ubuf(2, buf_terrain),
		_ubuf(3, buf_fr), _ubuf(4, buf_fl), _ubuf(5, buf_fd), _ubuf(6, buf_fu),
		_ubuf(7, sparse.buf_compact_tile_coords),
	], shader_flux, 0)
	uset_vel = rd.uniform_set_create([
		_ubuf(0, buf_fr), _ubuf(1, buf_fl), _ubuf(2, buf_fd), _ubuf(3, buf_fu),
		_ubuf(4, buf_terrain), _ubuf(5, buf_density),
		_ubuf(6, buf_gdir_x), _ubuf(7, buf_gdir_y),
		_ubuf(8, buf_out_vx), _ubuf(9, buf_out_vy),
		_ubuf(10, sparse.buf_compact_tile_coords),
		_ubuf(11, buf_ground_cost),
	], shader_vel, 0)


func _ubuf(binding: int, buf: RID) -> RDUniform:
	var u := RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u.binding = binding
	u.add_id(buf)
	return u


# ── Static data upload ──────────────────────────────────────────────────

func setup_bfs(faction_presence: RID, fac_to_group: RID) -> void:
	_bfs_presence = faction_presence
	_bfs_fac_to_group = fac_to_group
	uset_bfs = rd.uniform_set_create([
		_ubuf(0, faction_presence),
		_ubuf(1, buf_terrain),
		_ubuf(2, fac_to_group),
		_ubuf(3, buf_bfs_dist),
		_ubuf(4, buf_goal_dist_all),
		_ubuf(5, buf_goal_dist),
		_ubuf(6, buf_gdir_x),
		_ubuf(7, buf_gdir_y),
		_ubuf(8, buf_bfs_flags),
		_ubuf(9, buf_dir_lut),
		_ubuf(10, buf_ground_cost),
		_ubuf(11, buf_bfs_cost),
	], shader_bfs, 0)


const BFS_TILE := 16            # 与 goal_bfs.glsl 的 local_size 一致
const BFS_RELAX_PASSES := 80    # 动态场每帧迭代预算；收敛后 early-out。开局按批继续直到收敛，复杂加权迷宫可能需多批

func dispatch_goal_bfs(cl: int, target_group: int, num_factions: int) -> void:
	var gx := ceili(float(gw) / BFS_TILE)
	var gy := ceili(float(gh) / BFS_TILE)
	var p := PackedByteArray(); p.resize(32)
	p.encode_s32(0, gw)
	p.encode_s32(4, gh)
	p.encode_s32(12, target_group)
	p.encode_s32(16, num_factions)

	# INIT
	p.encode_s32(8, 0)
	rd.compute_list_bind_compute_pipeline(cl, pipeline_bfs)
	rd.compute_list_bind_uniform_set(cl, uset_bfs, 0)
	rd.compute_list_set_push_constant(cl, p, 32)
	rd.compute_list_dispatch(cl, gx, gy, 1)
	rd.compute_list_add_barrier(cl)

	# RELAX（分块松弛）：add_barrier 会自动重绑 pipeline / uniform set / push constant；relax_pass 用于轮换标志槽
	p.encode_s32(8, 1)
	for i in range(BFS_RELAX_PASSES):
		p.encode_s32(20, i)
		rd.compute_list_set_push_constant(cl, p, 32)
		rd.compute_list_dispatch(cl, gx, gy, 1)
		rd.compute_list_add_barrier(cl)

	# GRADIENT
	p.encode_s32(8, 2)
	rd.compute_list_set_push_constant(cl, p, 32)
	rd.compute_list_dispatch(cl, gx, gy, 1)


func upload_static_data() -> void:
	_upload(buf_terrain, terrain)
	_upload(buf_ground_cost, ground_cost)


func _upload(buf: RID, arr: PackedFloat32Array) -> void:
	var b := arr.to_byte_array()
	rd.buffer_update(buf, 0, b.size(), b)


# ── Dispatch (caller owns the compute list) ──────────────────────────────

func reset_compact_counter() -> void:
	sparse.reset_counter()


func dispatch_compact_tiles(cl: int) -> void:
	sparse.dispatch_compact(cl)


func dispatch_swe(cl: int, swe_dt: float) -> void:
	var push := PackedByteArray(); push.resize(48)
	push.encode_s32(0, gw)
	push.encode_s32(4, gh)
	push.encode_float(8, swe_dt)
	push.encode_float(12, cs)
	push.encode_float(16, gravity)
	push.encode_float(20, damping)
	push.encode_float(24, density_scale)
	push.encode_float(28, goal_scale)
	push.encode_float(32, wall_scale)
	push.encode_float(36, max_flux)
	push.encode_s32(40, 1 if sparse_enabled else 0)
	push.encode_float(44, 0.0)
	rd.compute_list_bind_compute_pipeline(cl, pipeline_flux)
	rd.compute_list_bind_uniform_set(cl, uset_flux, 0)
	rd.compute_list_set_push_constant(cl, push, push.size())
	if sparse_enabled:
		rd.compute_list_dispatch_indirect(cl, sparse.buf_indirect_args, 0)
	else:
		rd.compute_list_dispatch(cl, groups_x, groups_y, 1)


func dispatch_velocity(cl: int, num_groups: int = 1) -> void:
	var push := PackedByteArray(); push.resize(32)
	push.encode_s32(0, gw)
	push.encode_s32(4, gh)
	push.encode_float(8, agent_speed)
	push.encode_float(12, density_slowdown)
	push.encode_s32(16, 1 if sparse_enabled else 0)
	push.encode_s32(20, 0)
	push.encode_s32(24, 0)
	push.encode_s32(28, 0)
	rd.compute_list_bind_compute_pipeline(cl, pipeline_vel)
	rd.compute_list_bind_uniform_set(cl, uset_vel, 0)
	rd.compute_list_set_push_constant(cl, push, push.size())
	if sparse_enabled:
		rd.compute_list_dispatch_indirect(cl, sparse.buf_indirect_args, 0)
	else:
		rd.compute_list_dispatch(cl, groups_x, groups_y, num_groups)


# ── Readback ─────────────────────────────────────────────────────────────

func readback_compact_count() -> int:
	return sparse.readback_tile_count()


func readback_compact_tiles(count: int) -> PackedInt32Array:
	return sparse.readback_tile_coords(count)


func readback_density() -> void:
	density = rd.buffer_get_data(buf_density).to_float32_array()

func readback_velocity() -> void:
	var sz := cell_count * 4
	out_vx = rd.buffer_get_data(buf_out_vx, 0, sz).to_float32_array()
	out_vy = rd.buffer_get_data(buf_out_vy, 0, sz).to_float32_array()

func reset_flux() -> void:
	var bytes := _zero_bytes(cell_count)
	rd.buffer_update(buf_fr, 0, bytes.size(), bytes)
	rd.buffer_update(buf_fl, 0, bytes.size(), bytes)
	rd.buffer_update(buf_fd, 0, bytes.size(), bytes)
	rd.buffer_update(buf_fu, 0, bytes.size(), bytes)


func _zero_bytes(count: int) -> PackedByteArray:
	var a := PackedFloat32Array(); a.resize(count)
	return a.to_byte_array()


# ── Cleanup ──────────────────────────────────────────────────────────────

func cleanup() -> void:
	if rd == null:
		return
	# 先放 uniform set（引用 sparse / agents 的缓冲）；已被 Godot 连带释放的跳过
	for us in [uset_flux, uset_vel, uset_bfs]:
		if us.is_valid() and rd.uniform_set_is_valid(us): rd.free_rid(us)
	if sparse:
		sparse.cleanup()
		sparse = null
	for rid in [pipeline_flux, pipeline_vel, pipeline_bfs,
				shader_flux, shader_vel, shader_bfs,
				buf_density, buf_goal_dist, buf_goal_dist_all, buf_terrain, buf_ground_cost,
				buf_fr, buf_fl, buf_fd, buf_fu,
				buf_gdir_x, buf_gdir_y, buf_out_vx, buf_out_vy,
				buf_bfs_dist, buf_bfs_cost, buf_bfs_flags, buf_dir_lut]:
		if rid.is_valid():
			rd.free_rid(rid)
	rd = null
