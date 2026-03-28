class_name GPUAgents
extends RefCounted
## GPU-side agent simulation with counting-sort cell buffer.
##
## Owns: agent position/velocity (double-buffered), cell buffer.
## External shaders can bind cell_start / cell_count / sorted_idx
## for spatial neighbour queries.

var rd: RenderingDevice
var max_agents: int
var agent_count: int

var sg_w: int
var sg_h: int
var sg_cs: float
var sg_inv_cs: float
var num_cells: int

# Double-buffered agent data  [0]=ping [1]=pong
var buf_pos_x: Array[RID]
var buf_pos_y: Array[RID]
var buf_vel_x: Array[RID]
var buf_vel_y: Array[RID]
var cur: int = 0

# Cell buffer (public — other systems bind these)
var buf_cell_count: RID
var buf_cell_start: RID
var buf_cell_offset: RID
var buf_sorted_idx: RID
var buf_stall: RID

# Shaders / pipelines
var shd_clear: RID;   var pip_clear: RID
var shd_count: RID;   var pip_count: RID
var shd_prefix: RID;  var pip_prefix: RID
var shd_scatter: RID; var pip_scatter: RID
var shd_density: RID; var pip_density: RID
var shd_steer: RID;   var pip_steer: RID

# Uniform sets — indexed [cur] where applicable
var us_clear: RID
var us_prefix: RID
var us_count: Array[RID]
var us_scatter: Array[RID]
var us_density: Array[RID]
var us_steer: Array[RID]

var agent_groups: int
var cell_groups: int
var field_cell_groups: int
var frame_seed: int = 0

# Field buffer references (set via set_field_buffers)
var _buf_out_vx: RID
var _buf_out_vy: RID
var _buf_terrain: RID
var _buf_goal_dist: RID
var _buf_density: RID
var _field_gw: int
var _field_gh: int
var _field_cs: float


func _init(rendering_device: RenderingDevice, max_n: int, cell_size: float, world_size: Vector2) -> void:
	rd = rendering_device
	max_agents = max_n
	agent_count = 0
	sg_cs = cell_size
	sg_inv_cs = 1.0 / sg_cs
	sg_w = ceili(world_size.x / sg_cs) + 1
	sg_h = ceili(world_size.y / sg_cs) + 1
	num_cells = sg_w * sg_h

	agent_groups = ceili(float(max_agents) / 64.0)
	cell_groups  = ceili(float(num_cells) / 64.0)

	_create_buffers()
	_load_shaders()


func _create_buffers() -> void:
	var af := max_agents * 4
	var cf := num_cells * 4
	var zf := _zero_bytes(max_agents)
	var zc := _zero_bytes(num_cells)

	buf_pos_x = [rd.storage_buffer_create(af, zf), rd.storage_buffer_create(af, zf)]
	buf_pos_y = [rd.storage_buffer_create(af, zf), rd.storage_buffer_create(af, zf)]
	buf_vel_x = [rd.storage_buffer_create(af, zf), rd.storage_buffer_create(af, zf)]
	buf_vel_y = [rd.storage_buffer_create(af, zf), rd.storage_buffer_create(af, zf)]

	buf_cell_count  = rd.storage_buffer_create(cf, zc)
	buf_cell_start  = rd.storage_buffer_create(cf, zc)
	buf_cell_offset = rd.storage_buffer_create(cf, zc)
	buf_sorted_idx  = rd.storage_buffer_create(af, zf)
	buf_stall       = rd.storage_buffer_create(af, zf)


func _load_shaders() -> void:
	shd_clear   = _mk_shader("res://shaders/agent_clear_grid.glsl")
	shd_count   = _mk_shader("res://shaders/agent_count_cells.glsl")
	shd_prefix  = _mk_shader("res://shaders/agent_prefix_sum.glsl")
	shd_scatter = _mk_shader("res://shaders/agent_scatter.glsl")
	shd_density = _mk_shader("res://shaders/agent_density.glsl")
	shd_steer   = _mk_shader("res://shaders/agent_steer.glsl")
	pip_clear   = rd.compute_pipeline_create(shd_clear)
	pip_count   = rd.compute_pipeline_create(shd_count)
	pip_prefix  = rd.compute_pipeline_create(shd_prefix)
	pip_scatter = rd.compute_pipeline_create(shd_scatter)
	pip_density = rd.compute_pipeline_create(shd_density)
	pip_steer   = rd.compute_pipeline_create(shd_steer)


func _mk_shader(path: String) -> RID:
	return rd.shader_create_from_spirv((load(path) as RDShaderFile).get_spirv())


# ── Field buffer linking (call once, before build_uniform_sets) ─────────

func set_field_buffers(out_vx: RID, out_vy: RID, terrain: RID, goal_dist: RID,
					   density: RID, field_gw: int, field_gh: int, field_cs: float) -> void:
	_buf_out_vx = out_vx
	_buf_out_vy = out_vy
	_buf_terrain = terrain
	_buf_goal_dist = goal_dist
	_buf_density = density
	_field_gw = field_gw
	_field_gh = field_gh
	_field_cs = field_cs
	field_cell_groups = ceili(float(field_gw * field_gh) / 64.0)


func build_uniform_sets() -> void:
	us_clear = rd.uniform_set_create([_u(0, buf_cell_count)], shd_clear, 0)

	us_prefix = rd.uniform_set_create([
		_u(0, buf_cell_count), _u(1, buf_cell_start), _u(2, buf_cell_offset),
	], shd_prefix, 0)

	us_count   = [RID(), RID()]
	us_scatter = [RID(), RID()]
	us_density = [RID(), RID()]
	us_steer   = [RID(), RID()]

	for b in range(2):
		us_count[b] = rd.uniform_set_create([
			_u(0, buf_pos_x[b]), _u(1, buf_pos_y[b]), _u(2, buf_cell_count),
		], shd_count, 0)

		us_scatter[b] = rd.uniform_set_create([
			_u(0, buf_pos_x[b]), _u(1, buf_pos_y[b]),
			_u(2, buf_cell_offset), _u(3, buf_sorted_idx),
		], shd_scatter, 0)

		us_density[b] = rd.uniform_set_create([
			_u(0, buf_pos_x[b]), _u(1, buf_pos_y[b]),
			_u(2, buf_cell_start), _u(3, buf_cell_count),
			_u(4, buf_sorted_idx), _u(5, _buf_density),
		], shd_density, 0)

		var w := 1 - b
		us_steer[b] = rd.uniform_set_create([
			_u(0,  buf_pos_x[b]),  _u(1,  buf_pos_y[b]),
			_u(2,  buf_vel_x[b]),  _u(3,  buf_vel_y[b]),
			_u(4,  buf_pos_x[w]),  _u(5,  buf_pos_y[w]),
			_u(6,  buf_vel_x[w]),  _u(7,  buf_vel_y[w]),
			_u(8,  buf_cell_start), _u(9,  buf_cell_count),
			_u(10, buf_sorted_idx),
			_u(11, _buf_out_vx), _u(12, _buf_out_vy),
			_u(13, _buf_terrain), _u(14, _buf_goal_dist),
			_u(15, buf_stall),
		], shd_steer, 0)


func _u(binding: int, buf: RID) -> RDUniform:
	var u := RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u.binding = binding
	u.add_id(buf)
	return u


# ── Agent upload / readback ─────────────────────────────────────────────

func upload_agents(positions: PackedVector2Array, velocities: PackedVector2Array, count: int) -> void:
	agent_count = count
	var px := PackedFloat32Array(); px.resize(max_agents)
	var py := PackedFloat32Array(); py.resize(max_agents)
	var vx := PackedFloat32Array(); vx.resize(max_agents)
	var vy := PackedFloat32Array(); vy.resize(max_agents)
	for i in range(count):
		px[i] = positions[i].x;  py[i] = positions[i].y
		vx[i] = velocities[i].x; vy[i] = velocities[i].y
	_upf(buf_pos_x[cur], px); _upf(buf_pos_y[cur], py)
	_upf(buf_vel_x[cur], vx); _upf(buf_vel_y[cur], vy)


func readback_positions() -> PackedVector2Array:
	var w := 1 - cur
	var px := rd.buffer_get_data(buf_pos_x[w]).to_float32_array()
	var py := rd.buffer_get_data(buf_pos_y[w]).to_float32_array()
	var out := PackedVector2Array()
	out.resize(agent_count)
	for i in range(agent_count):
		out[i] = Vector2(px[i], py[i])
	cur = w
	return out


# ── Dispatches (append to an open compute list) ─────────────────────────

func dispatch_build_grid(cl: int) -> void:
	var pc16 := _push16_i(num_cells, 0, 0, 0)
	rd.compute_list_bind_compute_pipeline(cl, pip_clear)
	rd.compute_list_bind_uniform_set(cl, us_clear, 0)
	rd.compute_list_set_push_constant(cl, pc16, 16)
	rd.compute_list_dispatch(cl, cell_groups, 1, 1)
	rd.compute_list_add_barrier(cl)

	var pc_grid := _push_grid()
	rd.compute_list_bind_compute_pipeline(cl, pip_count)
	rd.compute_list_bind_uniform_set(cl, us_count[cur], 0)
	rd.compute_list_set_push_constant(cl, pc_grid, 16)
	rd.compute_list_dispatch(cl, agent_groups, 1, 1)
	rd.compute_list_add_barrier(cl)

	rd.compute_list_bind_compute_pipeline(cl, pip_prefix)
	rd.compute_list_bind_uniform_set(cl, us_prefix, 0)
	rd.compute_list_set_push_constant(cl, pc16, 16)
	rd.compute_list_dispatch(cl, 1, 1, 1)
	rd.compute_list_add_barrier(cl)

	rd.compute_list_bind_compute_pipeline(cl, pip_scatter)
	rd.compute_list_bind_uniform_set(cl, us_scatter[cur], 0)
	rd.compute_list_set_push_constant(cl, pc_grid, 16)
	rd.compute_list_dispatch(cl, agent_groups, 1, 1)


func dispatch_density(cl: int) -> void:
	var p := PackedByteArray(); p.resize(32)
	p.encode_s32(0,  _field_gw)
	p.encode_s32(4,  _field_gh)
	p.encode_float(8,  _field_cs)
	p.encode_float(12, 1.0 / _field_cs)
	p.encode_s32(16, sg_w)
	p.encode_s32(20, sg_h)
	p.encode_float(24, sg_cs)
	p.encode_s32(28, 0)
	rd.compute_list_bind_compute_pipeline(cl, pip_density)
	rd.compute_list_bind_uniform_set(cl, us_density[cur], 0)
	rd.compute_list_set_push_constant(cl, p, 32)
	rd.compute_list_dispatch(cl, field_cell_groups, 1, 1)


func dispatch_steer(cl: int, dt: float, sep_r: float, sep_str: float,
					steer_resp: float, world_w: float, world_h: float) -> void:
	var blend := clampf(steer_resp * dt, 0.0, 1.0)
	frame_seed += 7927

	var p := PackedByteArray(); p.resize(64)
	p.encode_s32(0,   agent_count)
	p.encode_s32(4,   sg_w)
	p.encode_s32(8,   sg_h)
	p.encode_float(12, sg_inv_cs)
	p.encode_float(16, dt)
	p.encode_float(20, blend)
	p.encode_float(24, sep_r * sep_r)
	p.encode_float(28, 1.0 / sep_r)
	p.encode_float(32, sep_str)
	p.encode_float(36, 1.0 / _field_cs)
	p.encode_s32(40,  _field_gw)
	p.encode_s32(44,  _field_gh)
	p.encode_float(48, world_w)
	p.encode_float(52, world_h)
	p.encode_float(56, _field_cs)
	p.encode_u32(60,  frame_seed)

	rd.compute_list_bind_compute_pipeline(cl, pip_steer)
	rd.compute_list_bind_uniform_set(cl, us_steer[cur], 0)
	rd.compute_list_set_push_constant(cl, p, 64)
	rd.compute_list_dispatch(cl, agent_groups, 1, 1)


# ── Helpers ──────────────────────────────────────────────────────────────

func _push_grid() -> PackedByteArray:
	var p := PackedByteArray(); p.resize(16)
	p.encode_s32(0, agent_count)
	p.encode_s32(4, sg_w)
	p.encode_s32(8, sg_h)
	p.encode_float(12, sg_inv_cs)
	return p


func _push16_i(a: int, b: int, c: int, d: int) -> PackedByteArray:
	var p := PackedByteArray(); p.resize(16)
	p.encode_s32(0, a); p.encode_s32(4, b); p.encode_s32(8, c); p.encode_s32(12, d)
	return p


func _upf(buf: RID, arr: PackedFloat32Array) -> void:
	var b := arr.to_byte_array()
	rd.buffer_update(buf, 0, b.size(), b)


func _zero_bytes(count: int) -> PackedByteArray:
	var a := PackedFloat32Array(); a.resize(count)
	return a.to_byte_array()


# ── Cleanup ──────────────────────────────────────────────────────────────

func cleanup() -> void:
	if rd == null:
		return
	for rid in [us_clear, us_prefix]:
		rd.free_rid(rid)
	for arr in [us_count, us_scatter, us_density, us_steer]:
		for rid in arr:
			rd.free_rid(rid)
	for pip in [pip_clear, pip_count, pip_prefix, pip_scatter, pip_density, pip_steer]:
		rd.free_rid(pip)
	for shd in [shd_clear, shd_count, shd_prefix, shd_scatter, shd_density, shd_steer]:
		rd.free_rid(shd)
	for arr in [buf_pos_x, buf_pos_y, buf_vel_x, buf_vel_y]:
		for rid in arr:
			rd.free_rid(rid)
	for rid in [buf_cell_count, buf_cell_start, buf_cell_offset, buf_sorted_idx, buf_stall]:
		rd.free_rid(rid)
	rd = null
