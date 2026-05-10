class_name GPUSWEField
extends RefCounted

var gw: int
var gh: int
var cs: float
var cell_count: int
var max_groups: int

# CPU-side arrays
var terrain: PackedFloat32Array
var goal_dist: PackedFloat32Array           # cell_count (flux shader only)
var goal_dist_all: PackedFloat32Array       # cell_count * max_groups (combat)
var density: PackedFloat32Array             # cell_count (readback)
var gdir_x: PackedFloat32Array              # cell_count * max_groups (velocity)
var gdir_y: PackedFloat32Array              # cell_count * max_groups (velocity)
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
var buf_goal_dist: RID
var buf_goal_dist_all: RID
var buf_terrain: RID
var buf_fr: RID
var buf_fl: RID
var buf_fd: RID
var buf_fu: RID
var buf_gdir_x: RID
var buf_gdir_y: RID
var buf_out_vx: RID
var buf_out_vy: RID
var buf_bfs_dist: RID
var uset_bfs: RID

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
	goal_dist     = PackedFloat32Array(); goal_dist.resize(cell_count); goal_dist.fill(1e6)
	goal_dist_all = PackedFloat32Array(); goal_dist_all.resize(cell_count * max_groups); goal_dist_all.fill(1e6)
	density       = PackedFloat32Array(); density.resize(cell_count)
	gdir_x        = PackedFloat32Array(); gdir_x.resize(cell_count * max_groups)
	gdir_y        = PackedFloat32Array(); gdir_y.resize(cell_count * max_groups)
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


func build_goal_field(goals: Array[Vector2i]) -> void:
	_bfs_into(goal_dist, goals)
	for i in range(cell_count):
		goal_dist_all[i] = goal_dist[i]
	_compute_gradient_for(goal_dist, 0)


func build_all_goal_fields(group_goals: Array) -> void:
	goal_dist_all.fill(1e6)
	if group_goals.is_empty():
		return
	build_goal_field_group(0, group_goals[0])
	var base_dist := goal_dist_all.slice(0, cell_count)
	var base_gx := gdir_x.slice(0, cell_count)
	var base_gy := gdir_y.slice(0, cell_count)
	var all_dist := PackedFloat32Array()
	var all_gx := PackedFloat32Array()
	var all_gy := PackedFloat32Array()
	for _g in range(max_groups):
		all_dist.append_array(base_dist)
		all_gx.append_array(base_gx)
		all_gy.append_array(base_gy)
	goal_dist_all = all_dist
	gdir_x = all_gx
	gdir_y = all_gy


func build_goal_field_group(group: int, goals) -> void:
	var tmp := PackedFloat32Array()
	tmp.resize(cell_count)
	tmp.fill(1e6)
	_bfs_into(tmp, goals)
	var off := group * cell_count
	for i in range(cell_count):
		goal_dist_all[off + i] = tmp[i]
	_compute_gradient_for(tmp, group)
	if group == 0:
		for i in range(cell_count):
			goal_dist[i] = tmp[i]


func _bfs_into(dist: PackedFloat32Array, goals) -> void:
	dist.fill(1e6)
	var q := PackedInt32Array()
	q.resize(cell_count * 2)
	var tail := 0
	for g in goals:
		var gv: Vector2i
		if g is Vector2i:
			gv = g
		else:
			gv = Vector2i(int(g.x), int(g.y))
		if gv.x >= 0 and gv.x < gw and gv.y >= 0 and gv.y < gh:
			var gi := idx(gv.x, gv.y)
			if terrain[gi] < 0.5 and dist[gi] > 0.5:
				dist[gi] = 0.0
				q[tail] = gi
				tail += 1
	var head := 0
	while head < tail:
		var ci := q[head]; head += 1
		var cx := ci % gw
		var cy := ci / gw
		var nd := dist[ci] + 1.0
		if cx > 0:
			var ni := ci - 1
			if terrain[ni] < 0.5 and nd < dist[ni]:
				dist[ni] = nd; q[tail] = ni; tail += 1
		if cx < gw - 1:
			var ni := ci + 1
			if terrain[ni] < 0.5 and nd < dist[ni]:
				dist[ni] = nd; q[tail] = ni; tail += 1
		if cy > 0:
			var ni := ci - gw
			if terrain[ni] < 0.5 and nd < dist[ni]:
				dist[ni] = nd; q[tail] = ni; tail += 1
		if cy < gh - 1:
			var ni := ci + gw
			if terrain[ni] < 0.5 and nd < dist[ni]:
				dist[ni] = nd; q[tail] = ni; tail += 1
	var max_d := 0.0
	for i in range(cell_count):
		if dist[i] < 1e5:
			max_d = maxf(max_d, dist[i])
	var fill_d := max_d + 1.0
	for i in range(cell_count):
		if dist[i] >= 1e5:
			dist[i] = fill_d


func _compute_gradient_for(dist: PackedFloat32Array, group: int) -> void:
	var off := group * cell_count
	var w := gw
	for y in range(gh):
		var row := y * w
		for x in range(w):
			var i := row + x
			if terrain[i] > 0.5:
				gdir_x[off + i] = 0.0; gdir_y[off + i] = 0.0
				continue
			var ddx := 0.0
			var ddy := 0.0
			if x > 0 and x < w - 1:
				ddx = dist[i + 1] - dist[i - 1]
			elif x == 0 and w > 1:
				ddx = dist[i + 1] - dist[i]
			elif x == w - 1 and w > 1:
				ddx = dist[i] - dist[i - 1]
			if y > 0 and y < gh - 1:
				ddy = dist[i + w] - dist[i - w]
			elif y == 0 and gh > 1:
				ddy = dist[i + w] - dist[i]
			elif y == gh - 1 and gh > 1:
				ddy = dist[i] - dist[i - w]
			var l := sqrt(ddx * ddx + ddy * ddy)
			if l > 1e-6:
				gdir_x[off + i] = -ddx / l
				gdir_y[off + i] = -ddy / l
			else:
				gdir_x[off + i] = 0.0; gdir_y[off + i] = 0.0


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
	], shader_vel, 0)


func _ubuf(binding: int, buf: RID) -> RDUniform:
	var u := RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u.binding = binding
	u.add_id(buf)
	return u


# ── Static data upload ──────────────────────────────────────────────────

func setup_bfs(faction_presence: RID, fac_to_group: RID) -> void:
	uset_bfs = rd.uniform_set_create([
		_ubuf(0, faction_presence),
		_ubuf(1, buf_terrain),
		_ubuf(2, fac_to_group),
		_ubuf(3, buf_bfs_dist),
		_ubuf(4, buf_goal_dist_all),
		_ubuf(5, buf_goal_dist),
		_ubuf(6, buf_gdir_x),
		_ubuf(7, buf_gdir_y),
	], shader_bfs, 0)


const BFS_RELAX_ITERS := 256

func dispatch_goal_bfs(cl: int, target_group: int, num_factions: int) -> void:
	var gx := ceili(float(gw) / 8.0)
	var gy := ceili(float(gh) / 8.0)
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

	# RELAX iterations
	p.encode_s32(8, 1)
	for _i in range(BFS_RELAX_ITERS):
		rd.compute_list_bind_compute_pipeline(cl, pipeline_bfs)
		rd.compute_list_bind_uniform_set(cl, uset_bfs, 0)
		rd.compute_list_set_push_constant(cl, p, 32)
		rd.compute_list_dispatch(cl, gx, gy, 1)
		rd.compute_list_add_barrier(cl)

	# GRADIENT
	p.encode_s32(8, 2)
	rd.compute_list_bind_compute_pipeline(cl, pipeline_bfs)
	rd.compute_list_bind_uniform_set(cl, uset_bfs, 0)
	rd.compute_list_set_push_constant(cl, p, 32)
	rd.compute_list_dispatch(cl, gx, gy, 1)
	rd.compute_list_add_barrier(cl)


func upload_static_data() -> void:
	_upload(buf_terrain, terrain)
	upload_goal_data()


func upload_goal_data() -> void:
	_upload(buf_goal_dist, goal_dist)
	var gx_bytes := gdir_x.to_byte_array()
	rd.buffer_update(buf_gdir_x, 0, gx_bytes.size(), gx_bytes)
	var gy_bytes := gdir_y.to_byte_array()
	rd.buffer_update(buf_gdir_y, 0, gy_bytes.size(), gy_bytes)
	var gda_bytes := goal_dist_all.to_byte_array()
	rd.buffer_update(buf_goal_dist_all, 0, gda_bytes.size(), gda_bytes)


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
	if sparse:
		sparse.cleanup()
		sparse = null
	for rid in [uset_flux, uset_vel, uset_bfs,
				pipeline_flux, pipeline_vel, pipeline_bfs,
				shader_flux, shader_vel, shader_bfs,
				buf_density, buf_goal_dist, buf_goal_dist_all, buf_terrain,
				buf_fr, buf_fl, buf_fd, buf_fu,
				buf_gdir_x, buf_gdir_y, buf_out_vx, buf_out_vy,
				buf_bfs_dist]:
		if rid.is_valid():
			rd.free_rid(rid)
	rd = null
