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

var shader_flux: RID
var shader_vel: RID
var pipeline_flux: RID
var pipeline_vel: RID
var uset_flux: RID
var uset_vel: RID

var groups_x: int
var groups_y: int


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
	var tmp := PackedFloat32Array()
	tmp.resize(cell_count)
	for g in range(mini(group_goals.size(), max_groups)):
		tmp.fill(1e6)
		_bfs_into(tmp, group_goals[g])
		var off := g * cell_count
		for i in range(cell_count):
			goal_dist_all[off + i] = tmp[i]
		_compute_gradient_for(tmp, g)
	for i in range(cell_count):
		goal_dist[i] = goal_dist_all[i]


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
	_upload(buf_terrain, terrain)
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


func dispatch_velocity(cl: int, num_groups: int = 1) -> void:
	var push := PackedByteArray(); push.resize(16)
	push.encode_s32(0, gw)
	push.encode_s32(4, gh)
	push.encode_float(8, agent_speed)
	push.encode_float(12, density_slowdown)
	rd.compute_list_bind_compute_pipeline(cl, pipeline_vel)
	rd.compute_list_bind_uniform_set(cl, uset_vel, 0)
	rd.compute_list_set_push_constant(cl, push, push.size())
	rd.compute_list_dispatch(cl, groups_x, groups_y, num_groups)


# ── Readback ─────────────────────────────────────────────────────────────

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
	rd.free_rid(uset_flux)
	rd.free_rid(uset_vel)
	rd.free_rid(pipeline_flux)
	rd.free_rid(pipeline_vel)
	rd.free_rid(shader_flux)
	rd.free_rid(shader_vel)
	for buf in [buf_density, buf_goal_dist, buf_goal_dist_all, buf_terrain,
				buf_fr, buf_fl, buf_fd, buf_fu,
				buf_gdir_x, buf_gdir_y, buf_out_vx, buf_out_vy]:
		rd.free_rid(buf)
	rd = null
