class_name GPUSWEField
extends RefCounted
## GPU-accelerated Shallow Water Equation velocity field.
##
## Accepts an external RenderingDevice (shared with GPUAgents).
## Density buffer is written by GPUAgents; this class only reads it.
## dispatch_swe / dispatch_velocity append to a caller-owned compute list.

var gw: int
var gh: int
var cs: float
var cell_count: int

# CPU-side arrays (static env + optional readback)
var terrain: PackedFloat32Array
var goal_dist: PackedFloat32Array
var density: PackedFloat32Array
var gdir_x: PackedFloat32Array
var gdir_y: PackedFloat32Array
var out_vx: PackedFloat32Array
var out_vy: PackedFloat32Array

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
var buf_terrain: RID
var buf_fr: RID
var buf_fl: RID
var buf_fd: RID
var buf_fu: RID
var buf_gdir_x: RID
var buf_gdir_y: RID
var buf_out_vx: RID
var buf_out_vy: RID

var shader_flux: RID
var shader_vel: RID
var pipeline_flux: RID
var pipeline_vel: RID
var uset_flux: RID
var uset_vel: RID

var groups_x: int
var groups_y: int


func _init(rendering_device: RenderingDevice, width: int, height: int, cell_size: float) -> void:
	rd = rendering_device
	gw = width
	gh = height
	cs = cell_size
	cell_count = gw * gh

	terrain   = PackedFloat32Array(); terrain.resize(cell_count)
	goal_dist = PackedFloat32Array(); goal_dist.resize(cell_count); goal_dist.fill(1e6)
	density   = PackedFloat32Array(); density.resize(cell_count)
	gdir_x    = PackedFloat32Array(); gdir_x.resize(cell_count)
	gdir_y    = PackedFloat32Array(); gdir_y.resize(cell_count)
	out_vx    = PackedFloat32Array(); out_vx.resize(cell_count)
	out_vy    = PackedFloat32Array(); out_vy.resize(cell_count)

	groups_x = ceili(float(gw) / 8.0)
	groups_y = ceili(float(gh) / 8.0)

	_init_gpu()


func idx(x: int, y: int) -> int:
	return y * gw + x


# ── Environment setup (CPU, one-time) ───────────────────────────────────

func add_wall(x0: int, y0: int, x1: int, y1: int) -> void:
	for yy in range(maxi(y0, 0), mini(y1, gh)):
		for xx in range(maxi(x0, 0), mini(x1, gw)):
			terrain[idx(xx, yy)] = 1.0


func build_goal_field(goals: Array[Vector2i]) -> void:
	goal_dist.fill(1e6)
	var q: Array[int] = []
	for g in goals:
		if g.x >= 0 and g.x < gw and g.y >= 0 and g.y < gh:
			if terrain[idx(g.x, g.y)] < 0.5:
				goal_dist[idx(g.x, g.y)] = 0.0
				q.append(idx(g.x, g.y))
	var head := 0
	var DX: Array[int] = [1, -1, 0, 0]
	var DY: Array[int] = [0, 0, 1, -1]
	while head < q.size():
		var ci := q[head]; head += 1
		var cx := ci % gw
		var cy := ci / gw
		var cd := goal_dist[ci]
		for d in range(4):
			var nx := cx + DX[d]
			var ny := cy + DY[d]
			if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
				continue
			var ni := idx(nx, ny)
			if terrain[ni] > 0.5:
				continue
			if cd + 1.0 < goal_dist[ni]:
				goal_dist[ni] = cd + 1.0
				q.append(ni)
	var max_d := 0.0
	for i in range(cell_count):
		if goal_dist[i] < 1e5:
			max_d = maxf(max_d, goal_dist[i])
	for i in range(cell_count):
		if goal_dist[i] >= 1e5:
			goal_dist[i] = max_d + 1.0
	_compute_goal_gradient()


func _compute_goal_gradient() -> void:
	for y in range(gh):
		for x in range(gw):
			var i := idx(x, y)
			if terrain[i] > 0.5:
				gdir_x[i] = 0.0; gdir_y[i] = 0.0
				continue
			var ddx := 0.0
			var ddy := 0.0
			if x > 0 and x < gw - 1:
				ddx = goal_dist[idx(x + 1, y)] - goal_dist[idx(x - 1, y)]
			elif x == 0 and gw > 1:
				ddx = goal_dist[idx(1, y)] - goal_dist[i]
			elif x == gw - 1 and gw > 1:
				ddx = goal_dist[i] - goal_dist[idx(gw - 2, y)]
			if y > 0 and y < gh - 1:
				ddy = goal_dist[idx(x, y + 1)] - goal_dist[idx(x, y - 1)]
			elif y == 0 and gh > 1:
				ddy = goal_dist[idx(x, 1)] - goal_dist[i]
			elif y == gh - 1 and gh > 1:
				ddy = goal_dist[i] - goal_dist[idx(x, gh - 2)]
			var l := sqrt(ddx * ddx + ddy * ddy)
			if l > 1e-6:
				gdir_x[i] = -ddx / l
				gdir_y[i] = -ddy / l
			else:
				gdir_x[i] = 0.0; gdir_y[i] = 0.0


# ── GPU init ─────────────────────────────────────────────────────────────

func _init_gpu() -> void:
	var n_bytes := cell_count * 4
	var zb := _zero_bytes(cell_count)

	buf_density   = rd.storage_buffer_create(n_bytes, zb)
	buf_goal_dist = rd.storage_buffer_create(n_bytes, zb)
	buf_terrain   = rd.storage_buffer_create(n_bytes, zb)
	buf_fr        = rd.storage_buffer_create(n_bytes, zb)
	buf_fl        = rd.storage_buffer_create(n_bytes, zb)
	buf_fd        = rd.storage_buffer_create(n_bytes, zb)
	buf_fu        = rd.storage_buffer_create(n_bytes, zb)
	buf_gdir_x    = rd.storage_buffer_create(n_bytes, zb)
	buf_gdir_y    = rd.storage_buffer_create(n_bytes, zb)
	buf_out_vx    = rd.storage_buffer_create(n_bytes, zb)
	buf_out_vy    = rd.storage_buffer_create(n_bytes, zb)

	var flux_spirv := (load("res://shaders/swe_flux.glsl") as RDShaderFile).get_spirv()
	shader_flux   = rd.shader_create_from_spirv(flux_spirv)
	pipeline_flux = rd.compute_pipeline_create(shader_flux)

	var vel_spirv := (load("res://shaders/swe_velocity.glsl") as RDShaderFile).get_spirv()
	shader_vel   = rd.shader_create_from_spirv(vel_spirv)
	pipeline_vel = rd.compute_pipeline_create(shader_vel)

	_build_uniform_sets()


func _build_uniform_sets() -> void:
	uset_flux = rd.uniform_set_create([
		_ubuf(0, buf_density), _ubuf(1, buf_goal_dist), _ubuf(2, buf_terrain),
		_ubuf(3, buf_fr), _ubuf(4, buf_fl), _ubuf(5, buf_fd), _ubuf(6, buf_fu),
	], shader_flux, 0)
	uset_vel = rd.uniform_set_create([
		_ubuf(0, buf_fr), _ubuf(1, buf_fl), _ubuf(2, buf_fd), _ubuf(3, buf_fu),
		_ubuf(4, buf_terrain), _ubuf(5, buf_density),
		_ubuf(6, buf_gdir_x), _ubuf(7, buf_gdir_y),
		_ubuf(8, buf_out_vx), _ubuf(9, buf_out_vy),
	], shader_vel, 0)


func _ubuf(binding: int, buf: RID) -> RDUniform:
	var u := RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u.binding = binding
	u.add_id(buf)
	return u


# ── Static data upload ──────────────────────────────────────────────────

func upload_static_data() -> void:
	_upload(buf_terrain,   terrain)
	_upload(buf_goal_dist, goal_dist)
	_upload(buf_gdir_x,    gdir_x)
	_upload(buf_gdir_y,    gdir_y)


func _upload(buf: RID, arr: PackedFloat32Array) -> void:
	var b := arr.to_byte_array()
	rd.buffer_update(buf, 0, b.size(), b)


# ── Dispatch (caller owns the compute list) ──────────────────────────────

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
	rd.compute_list_bind_compute_pipeline(cl, pipeline_flux)
	rd.compute_list_bind_uniform_set(cl, uset_flux, 0)
	rd.compute_list_set_push_constant(cl, push, push.size())
	rd.compute_list_dispatch(cl, groups_x, groups_y, 1)


func dispatch_velocity(cl: int) -> void:
	var push := PackedByteArray(); push.resize(16)
	push.encode_s32(0, gw)
	push.encode_s32(4, gh)
	push.encode_float(8, agent_speed)
	push.encode_float(12, density_slowdown)
	rd.compute_list_bind_compute_pipeline(cl, pipeline_vel)
	rd.compute_list_bind_uniform_set(cl, uset_vel, 0)
	rd.compute_list_set_push_constant(cl, push, push.size())
	rd.compute_list_dispatch(cl, groups_x, groups_y, 1)


# ── Optional readback (for overlays) ────────────────────────────────────

func readback_density() -> void:
	density = rd.buffer_get_data(buf_density).to_float32_array()

func readback_velocity() -> void:
	out_vx = rd.buffer_get_data(buf_out_vx).to_float32_array()
	out_vy = rd.buffer_get_data(buf_out_vy).to_float32_array()


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
	rd.free_rid(uset_flux)
	rd.free_rid(uset_vel)
	rd.free_rid(pipeline_flux)
	rd.free_rid(pipeline_vel)
	rd.free_rid(shader_flux)
	rd.free_rid(shader_vel)
	for buf in [buf_density, buf_goal_dist, buf_terrain,
				buf_fr, buf_fl, buf_fd, buf_fu,
				buf_gdir_x, buf_gdir_y, buf_out_vx, buf_out_vy]:
		rd.free_rid(buf)
	rd = null
