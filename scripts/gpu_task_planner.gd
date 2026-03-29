class_name GPUTaskPlanner
extends RefCounted
## GPU-parallel task planner using auction-based assignment.
##
## Pipeline per frame:
##   1. Clear task claim buffer (CPU upload)
##   2. task_score   — each agent scores all tasks, atomicMin bid
##   3. task_assign  — check auction winners, update assignments
##   4. task_navigate — steer agents toward assigned task positions
##   5. task_complete — arrival check, stamina effects, task status

var rd: RenderingDevice
var max_agents: int
var max_tasks: int
var agent_count: int
var task_count: int

var grid_w: int
var grid_h: int
var cell_size: float
var world_w: float
var world_h: float

# Double-buffered agent position / velocity  [0]=ping [1]=pong
var buf_pos_x: Array[RID]
var buf_pos_y: Array[RID]
var buf_vel_x: Array[RID]
var buf_vel_y: Array[RID]
var cur: int = 0

# Agent extended state
var buf_stamina: RID
var buf_target: RID

# Task data
var buf_task_pos_x: RID
var buf_task_pos_y: RID
var buf_task_cost: RID
var buf_task_priority: RID
var buf_task_status: RID
var buf_task_owner: RID
var buf_task_type: RID

# Auction intermediates
var buf_best_task: RID
var buf_best_bid: RID
var buf_task_claim: RID

# Terrain
var buf_terrain: RID

# Stall detection
var buf_stall: RID

# BFS flow field (per-task distance fields)
const MAX_SLOTS := 64
var cell_count: int
var buf_goal_dist_all: RID
var buf_task_slot: RID

# Shaders + pipelines
var shd_score: RID;    var pip_score: RID
var shd_assign: RID;   var pip_assign: RID
var shd_navigate: RID; var pip_navigate: RID
var shd_complete: RID; var pip_complete: RID

# Uniform sets  — indexed [cur] where applicable
var us_score: Array[RID]
var us_assign: RID
var us_navigate: Array[RID]
var us_complete: Array[RID]

var claim_clear_bytes: PackedByteArray
var frame_seed: int = 0

# Tuning
var nav_speed := 70.0
var sep_radius := 16.0
var sep_strength := 100.0
var steer_response := 6.0
var arrival_radius := 15.0
var stamina_decay := 0.02


func _init(rendering_device: RenderingDevice, max_n: int, max_t: int,
			gw: int, gh: int, cs: float) -> void:
	rd = rendering_device
	max_agents = max_n
	max_tasks = max_t
	agent_count = 0
	task_count = 0
	grid_w = gw
	grid_h = gh
	cell_size = cs
	world_w = float(gw) * cs
	world_h = float(gh) * cs
	cell_count = gw * gh

	claim_clear_bytes = PackedByteArray()
	claim_clear_bytes.resize(max_tasks * 4)
	claim_clear_bytes.fill(0xFF)

	_create_buffers()
	_load_shaders()


func _create_buffers() -> void:
	var af := max_agents * 4
	var tf := max_tasks * 4
	var za := _zero_bytes(max_agents)
	var zt := _zero_bytes(max_tasks)
	var zg := _zero_bytes(grid_w * grid_h)

	buf_pos_x = [rd.storage_buffer_create(af, za), rd.storage_buffer_create(af, za)]
	buf_pos_y = [rd.storage_buffer_create(af, za), rd.storage_buffer_create(af, za)]
	buf_vel_x = [rd.storage_buffer_create(af, za), rd.storage_buffer_create(af, za)]
	buf_vel_y = [rd.storage_buffer_create(af, za), rd.storage_buffer_create(af, za)]

	buf_stamina = rd.storage_buffer_create(af, za)
	var target_init := PackedByteArray()
	target_init.resize(max_agents * 4)
	target_init.fill(0xFF)
	buf_target = rd.storage_buffer_create(af, target_init)

	buf_task_pos_x    = rd.storage_buffer_create(tf, zt)
	buf_task_pos_y    = rd.storage_buffer_create(tf, zt)
	buf_task_cost     = rd.storage_buffer_create(tf, zt)
	buf_task_priority = rd.storage_buffer_create(tf, zt)
	buf_task_status   = rd.storage_buffer_create(tf, zt)
	buf_task_owner    = rd.storage_buffer_create(tf, zt)
	buf_task_type     = rd.storage_buffer_create(tf, zt)

	buf_best_task  = rd.storage_buffer_create(af, za)
	buf_best_bid   = rd.storage_buffer_create(af, za)
	buf_task_claim = rd.storage_buffer_create(tf, zt)

	buf_terrain = rd.storage_buffer_create(grid_w * grid_h * 4, zg)
	buf_stall   = rd.storage_buffer_create(af, za)

	var gd_bytes := MAX_SLOTS * cell_count * 4
	var zd := PackedByteArray(); zd.resize(gd_bytes)
	var big_val := PackedFloat32Array([1e6])
	var bv := big_val.to_byte_array()
	for k in range(MAX_SLOTS * cell_count):
		zd[k * 4]     = bv[0]; zd[k * 4 + 1] = bv[1]
		zd[k * 4 + 2] = bv[2]; zd[k * 4 + 3] = bv[3]
	buf_goal_dist_all = rd.storage_buffer_create(gd_bytes, zd)

	var slot_init := PackedInt32Array(); slot_init.resize(max_tasks)
	slot_init.fill(-1)
	buf_task_slot = rd.storage_buffer_create(max_tasks * 4, slot_init.to_byte_array())


func _load_shaders() -> void:
	shd_score    = _mk_shader("res://shaders/task_score.glsl")
	shd_assign   = _mk_shader("res://shaders/task_assign.glsl")
	shd_navigate = _mk_shader("res://shaders/task_navigate.glsl")
	shd_complete = _mk_shader("res://shaders/task_complete.glsl")
	pip_score    = rd.compute_pipeline_create(shd_score)
	pip_assign   = rd.compute_pipeline_create(shd_assign)
	pip_navigate = rd.compute_pipeline_create(shd_navigate)
	pip_complete = rd.compute_pipeline_create(shd_complete)


func _mk_shader(path: String) -> RID:
	return rd.shader_create_from_spirv((load(path) as RDShaderFile).get_spirv())


# ── Uniform sets ─────────────────────────────────────────────────────────

func build_uniform_sets() -> void:
	us_score    = [RID(), RID()]
	us_navigate = [RID(), RID()]
	us_complete = [RID(), RID()]

	for b in range(2):
		var w := 1 - b
		us_score[b] = rd.uniform_set_create([
			_u(0,  buf_pos_x[b]),     _u(1,  buf_pos_y[b]),
			_u(2,  buf_stamina),      _u(3,  buf_target),
			_u(4,  buf_task_pos_x),   _u(5,  buf_task_pos_y),
			_u(6,  buf_task_cost),    _u(7,  buf_task_priority),
			_u(8,  buf_task_status),  _u(9,  buf_task_type),
			_u(10, buf_best_task),    _u(11, buf_best_bid),
			_u(12, buf_task_claim),
		], shd_score, 0)

		us_navigate[b] = rd.uniform_set_create([
			_u(0,  buf_pos_x[b]),  _u(1,  buf_pos_y[b]),
			_u(2,  buf_vel_x[b]),  _u(3,  buf_vel_y[b]),
			_u(4,  buf_pos_x[w]),  _u(5,  buf_pos_y[w]),
			_u(6,  buf_vel_x[w]),  _u(7,  buf_vel_y[w]),
			_u(8,  buf_target),
			_u(9,  buf_task_pos_x), _u(10, buf_task_pos_y),
			_u(11, buf_terrain),
			_u(12, buf_stall),
			_u(13, buf_task_status),
			_u(14, buf_goal_dist_all),
			_u(15, buf_task_slot),
		], shd_navigate, 0)

		us_complete[b] = rd.uniform_set_create([
			_u(0, buf_pos_x[w]),  _u(1, buf_pos_y[w]),
			_u(2, buf_stamina),   _u(3, buf_target),
			_u(4, buf_task_pos_x), _u(5, buf_task_pos_y),
			_u(6, buf_task_cost), _u(7, buf_task_status),
			_u(8, buf_task_type),
		], shd_complete, 0)

	us_assign = rd.uniform_set_create([
		_u(0, buf_best_task), _u(1, buf_best_bid),
		_u(2, buf_task_claim),
		_u(3, buf_target),    _u(4, buf_task_status),
		_u(5, buf_task_owner),
	], shd_assign, 0)


func _u(binding: int, buf: RID) -> RDUniform:
	var u := RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u.binding = binding
	u.add_id(buf)
	return u


# ── Upload ───────────────────────────────────────────────────────────────

func upload_agents(positions: PackedVector2Array, velocities: PackedVector2Array,
				   stamina: PackedFloat32Array, count: int) -> void:
	agent_count = count
	var px := PackedFloat32Array(); px.resize(max_agents)
	var py := PackedFloat32Array(); py.resize(max_agents)
	var vx := PackedFloat32Array(); vx.resize(max_agents)
	var vy := PackedFloat32Array(); vy.resize(max_agents)
	var st := PackedFloat32Array(); st.resize(max_agents)
	for i in range(count):
		px[i] = positions[i].x;  py[i] = positions[i].y
		vx[i] = velocities[i].x; vy[i] = velocities[i].y
		st[i] = stamina[i]
	_upf(buf_pos_x[cur], px); _upf(buf_pos_y[cur], py)
	_upf(buf_vel_x[cur], vx); _upf(buf_vel_y[cur], vy)
	_upf(buf_stamina, st)

	var tgt := PackedByteArray()
	tgt.resize(max_agents * 4)
	tgt.fill(0xFF)
	rd.buffer_update(buf_target, 0, tgt.size(), tgt)


func upload_tasks(pos_x: PackedFloat32Array, pos_y: PackedFloat32Array,
				  cost: PackedFloat32Array, priority: PackedFloat32Array,
				  type: PackedInt32Array, status: PackedInt32Array, count: int) -> void:
	task_count = count
	_upf(buf_task_pos_x, pos_x)
	_upf(buf_task_pos_y, pos_y)
	_upf(buf_task_cost, cost)
	_upf(buf_task_priority, priority)
	_upu(buf_task_type, type)
	_upu(buf_task_status, status)

	var owner := PackedByteArray()
	owner.resize(max_tasks * 4)
	owner.fill(0xFF)
	rd.buffer_update(buf_task_owner, 0, owner.size(), owner)


func upload_terrain(terrain: PackedFloat32Array) -> void:
	_upf(buf_terrain, terrain)


func upload_goal_dist_slot(slot: int, data: PackedFloat32Array) -> void:
	var offset := slot * cell_count * 4
	var b := data.to_byte_array()
	rd.buffer_update(buf_goal_dist_all, offset, b.size(), b)


func upload_task_slot_map(map: PackedInt32Array) -> void:
	_upu(buf_task_slot, map)


func update_tasks_partial(indices: PackedInt32Array,
						  pos_x: PackedFloat32Array, pos_y: PackedFloat32Array,
						  cost: PackedFloat32Array, priority: PackedFloat32Array,
						  type: PackedInt32Array) -> void:
	for k in range(indices.size()):
		var idx := indices[k]
		var off := idx * 4
		rd.buffer_update(buf_task_pos_x, off, 4, _f2b(pos_x[k]))
		rd.buffer_update(buf_task_pos_y, off, 4, _f2b(pos_y[k]))
		rd.buffer_update(buf_task_cost, off, 4, _f2b(cost[k]))
		rd.buffer_update(buf_task_priority, off, 4, _f2b(priority[k]))
		rd.buffer_update(buf_task_type, off, 4, _i2b(type[k]))
		rd.buffer_update(buf_task_status, off, 4, _i2b(0))
		rd.buffer_update(buf_task_owner, off, 4, _i2b(-1))


# ── Dispatch ─────────────────────────────────────────────────────────────

func clear_claims() -> void:
	rd.buffer_update(buf_task_claim, 0, task_count * 4, claim_clear_bytes.slice(0, task_count * 4))


func dispatch_score(cl: int) -> void:
	var p := PackedByteArray(); p.resize(16)
	p.encode_s32(0, agent_count)
	p.encode_s32(4, task_count)
	p.encode_s32(8, 0)
	p.encode_s32(12, 0)
	rd.compute_list_bind_compute_pipeline(cl, pip_score)
	rd.compute_list_bind_uniform_set(cl, us_score[cur], 0)
	rd.compute_list_set_push_constant(cl, p, 16)
	rd.compute_list_dispatch(cl, _agent_groups(), 1, 1)


func dispatch_assign(cl: int) -> void:
	var p := PackedByteArray(); p.resize(16)
	p.encode_s32(0, agent_count)
	p.encode_s32(4, task_count)
	p.encode_s32(8, 0)
	p.encode_s32(12, 0)
	rd.compute_list_bind_compute_pipeline(cl, pip_assign)
	rd.compute_list_bind_uniform_set(cl, us_assign, 0)
	rd.compute_list_set_push_constant(cl, p, 16)
	rd.compute_list_dispatch(cl, _agent_groups(), 1, 1)


func dispatch_navigate(cl: int, dt: float) -> void:
	var blend := clampf(steer_response * dt, 0.0, 1.0)
	frame_seed += 7927

	var p := PackedByteArray(); p.resize(64)
	p.encode_s32(0,   agent_count)
	p.encode_float(4,  dt)
	p.encode_float(8,  blend)
	p.encode_float(12, sep_radius * sep_radius)
	p.encode_float(16, sep_strength)
	p.encode_float(20, 1.0 / sep_radius)
	p.encode_float(24, world_w)
	p.encode_float(28, world_h)
	p.encode_s32(32,  grid_w)
	p.encode_s32(36,  grid_h)
	p.encode_float(40, cell_size)
	p.encode_float(44, 1.0 / cell_size)
	p.encode_u32(48,  frame_seed)
	p.encode_float(52, nav_speed)
	p.encode_s32(56, cell_count)
	p.encode_s32(60, 0)

	rd.compute_list_bind_compute_pipeline(cl, pip_navigate)
	rd.compute_list_bind_uniform_set(cl, us_navigate[cur], 0)
	rd.compute_list_set_push_constant(cl, p, 64)
	rd.compute_list_dispatch(cl, _agent_groups(), 1, 1)


func dispatch_complete(cl: int, dt: float) -> void:
	var p := PackedByteArray(); p.resize(32)
	p.encode_s32(0,   agent_count)
	p.encode_s32(4,   task_count)
	p.encode_float(8,  dt)
	p.encode_float(12, arrival_radius * arrival_radius)
	p.encode_float(16, stamina_decay)
	p.encode_s32(20, 0)
	p.encode_s32(24, 0)
	p.encode_s32(28, 0)

	rd.compute_list_bind_compute_pipeline(cl, pip_complete)
	rd.compute_list_bind_uniform_set(cl, us_complete[cur], 0)
	rd.compute_list_set_push_constant(cl, p, 32)
	rd.compute_list_dispatch(cl, _agent_groups(), 1, 1)


# ── Readback ─────────────────────────────────────────────────────────────

func readback_positions() -> PackedVector2Array:
	var w := 1 - cur
	var px := rd.buffer_get_data(buf_pos_x[w]).to_float32_array()
	var py := rd.buffer_get_data(buf_pos_y[w]).to_float32_array()
	var out := PackedVector2Array(); out.resize(agent_count)
	for i in range(agent_count):
		out[i] = Vector2(px[i], py[i])
	cur = w
	return out


func readback_stamina() -> PackedFloat32Array:
	return rd.buffer_get_data(buf_stamina).to_float32_array()


func readback_targets() -> PackedInt32Array:
	return rd.buffer_get_data(buf_target).to_int32_array()


func readback_task_status() -> PackedInt32Array:
	return rd.buffer_get_data(buf_task_status).to_int32_array()


# ── Helpers ──────────────────────────────────────────────────────────────

func _agent_groups() -> int:
	return maxi(ceili(float(agent_count) / 64.0), 1)


func _upf(buf: RID, arr: PackedFloat32Array) -> void:
	var b := arr.to_byte_array()
	rd.buffer_update(buf, 0, b.size(), b)


func _upu(buf: RID, arr: PackedInt32Array) -> void:
	var b := arr.to_byte_array()
	rd.buffer_update(buf, 0, b.size(), b)


func _f2b(val: float) -> PackedByteArray:
	var a := PackedFloat32Array([val])
	return a.to_byte_array()


func _i2b(val: int) -> PackedByteArray:
	var a := PackedInt32Array([val])
	return a.to_byte_array()


func _zero_bytes(count: int) -> PackedByteArray:
	var a := PackedFloat32Array(); a.resize(count)
	return a.to_byte_array()


# ── Cleanup ──────────────────────────────────────────────────────────────

func cleanup() -> void:
	if rd == null:
		return
	for us in [us_assign]:
		rd.free_rid(us)
	for arr in [us_score, us_navigate, us_complete]:
		for rid in arr:
			rd.free_rid(rid)
	for pip in [pip_score, pip_assign, pip_navigate, pip_complete]:
		rd.free_rid(pip)
	for shd in [shd_score, shd_assign, shd_navigate, shd_complete]:
		rd.free_rid(shd)
	for arr in [buf_pos_x, buf_pos_y, buf_vel_x, buf_vel_y]:
		for rid in arr:
			rd.free_rid(rid)
	for rid in [buf_stamina, buf_target,
				buf_task_pos_x, buf_task_pos_y, buf_task_cost,
				buf_task_priority, buf_task_status, buf_task_owner, buf_task_type,
				buf_best_task, buf_best_bid, buf_task_claim,
				buf_terrain, buf_stall,
				buf_goal_dist_all, buf_task_slot]:
		rd.free_rid(rid)
	rd = null
