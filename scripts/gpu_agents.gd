class_name GPUAgents
extends RefCounted

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

# Cell buffer
var buf_cell_count: RID
var buf_cell_start: RID
var buf_cell_offset: RID
var buf_sorted_idx: RID
var buf_stall: RID

# Combat per-agent buffers
var buf_agent_info: RID
var buf_damage_acc: RID
var buf_max_hp: RID
var buf_regen_rate: RID
var buf_cooldown: RID
var buf_atk_range: RID
var buf_atk_damage: RID

# Per-cell combat buffer
var buf_faction_presence: RID
var buf_corpse_map: RID
var buf_cell_attacker: RID
var buf_cell_blocked: RID
var buf_disp_vx: RID
var buf_disp_vy: RID

# Small tables (32 entries each)
var buf_alliance: RID
var buf_fac_to_group: RID

# Shaders / pipelines
var shd_clear: RID;        var pip_clear: RID
var shd_count: RID;        var pip_count: RID
var shd_prefix: RID;       var pip_prefix: RID
var shd_scatter: RID;      var pip_scatter: RID
var shd_density: RID;      var pip_density: RID
var shd_steer: RID;        var pip_steer: RID
var shd_combat: RID;       var pip_combat: RID
var shd_cell_blocked: RID; var pip_cell_blocked: RID
var shd_disp_flow: RID;   var pip_disp_flow: RID

# Uniform sets
var us_clear: RID
var us_prefix: RID
var us_count: Array[RID]
var us_scatter: Array[RID]
var us_density: Array[RID]
var us_steer: Array[RID]
var us_steer_s1: RID
var us_combat: Array[RID]
var us_cell_blocked: RID
var us_disp_flow: RID

var agent_groups: int
var cell_groups: int
var field_cell_groups: int
var frame_seed: int = 0

# Field buffer references
var _buf_out_vx: RID
var _buf_out_vy: RID
var _buf_terrain: RID
var _buf_goal_dist: RID
var _buf_goal_dist_all: RID
var _buf_density: RID
var _field_gw: int
var _field_gh: int
var _field_cs: float
var _field_cell_count: int
var _max_groups: int


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

	# Combat per-agent
	buf_agent_info = rd.storage_buffer_create(af, zf)
	buf_damage_acc = rd.storage_buffer_create(af, zf)
	buf_max_hp     = rd.storage_buffer_create(af, zf)
	buf_regen_rate = rd.storage_buffer_create(af, zf)
	buf_cooldown   = rd.storage_buffer_create(af, zf)
	buf_atk_range  = rd.storage_buffer_create(af, zf)
	buf_atk_damage = rd.storage_buffer_create(af, zf)

	# Small tables (32 uint each = 128 bytes)
	var z32 := _zero_bytes(32)
	buf_alliance     = rd.storage_buffer_create(128, z32)
	buf_fac_to_group = rd.storage_buffer_create(128, z32)


func _load_shaders() -> void:
	shd_clear   = _mk_shader("res://shaders/agent_clear_grid.glsl")
	shd_count   = _mk_shader("res://shaders/agent_count_cells.glsl")
	shd_prefix  = _mk_shader("res://shaders/agent_prefix_sum.glsl")
	shd_scatter = _mk_shader("res://shaders/agent_scatter.glsl")
	shd_density = _mk_shader("res://shaders/agent_density.glsl")
	shd_steer   = _mk_shader("res://shaders/agent_steer.glsl")
	shd_combat       = _mk_shader("res://shaders/agent_combat.glsl")
	shd_cell_blocked = _mk_shader("res://shaders/cell_blocked.glsl")
	shd_disp_flow    = _mk_shader("res://shaders/disp_flow.glsl")
	pip_clear   = rd.compute_pipeline_create(shd_clear)
	pip_count   = rd.compute_pipeline_create(shd_count)
	pip_prefix  = rd.compute_pipeline_create(shd_prefix)
	pip_scatter = rd.compute_pipeline_create(shd_scatter)
	pip_density = rd.compute_pipeline_create(shd_density)
	pip_steer   = rd.compute_pipeline_create(shd_steer)
	pip_combat       = rd.compute_pipeline_create(shd_combat)
	pip_cell_blocked = rd.compute_pipeline_create(shd_cell_blocked)
	pip_disp_flow    = rd.compute_pipeline_create(shd_disp_flow)


func _mk_shader(path: String) -> RID:
	return rd.shader_create_from_spirv((load(path) as RDShaderFile).get_spirv())


# ── Field buffer linking ─────────────────────────────────────────────────

func set_field_buffers(out_vx: RID, out_vy: RID, terrain: RID, goal_dist: RID,
					   density: RID, field_gw: int, field_gh: int, field_cs: float,
					   goal_dist_all: RID = RID(), max_groups: int = 2) -> void:
	_buf_out_vx = out_vx
	_buf_out_vy = out_vy
	_buf_terrain = terrain
	_buf_goal_dist = goal_dist
	_buf_goal_dist_all = goal_dist_all
	_buf_density = density
	_field_gw = field_gw
	_field_gh = field_gh
	_field_cs = field_cs
	_field_cell_count = field_gw * field_gh
	_max_groups = max_groups
	field_cell_groups = ceili(float(_field_cell_count) / 64.0)

	# Create faction_presence buffer (needs field dimensions)
	var zfc := _zero_bytes(_field_cell_count)
	buf_faction_presence = rd.storage_buffer_create(_field_cell_count * 4, zfc)

	# Corpse map: 0xFFFFFFFF = empty, agent index = corpse placed
	var ffc := PackedByteArray()
	ffc.resize(_field_cell_count * 4)
	ffc.fill(0xFF)
	buf_corpse_map = rd.storage_buffer_create(_field_cell_count * 4, ffc)

	# Cell attacker map: 0xFFFFFFFF = empty, agent index = attacker occupying cell
	var fca := PackedByteArray()
	fca.resize(_field_cell_count * 4)
	fca.fill(0xFF)
	buf_cell_attacker = rd.storage_buffer_create(_field_cell_count * 4, fca)

	# Cell blocked map: per-cell bitmask of blocked alliance groups
	buf_cell_blocked = rd.storage_buffer_create(_field_cell_count * 4, zfc)

	# DISP flow field: per-cell * max_groups velocity
	var zfg := _zero_bytes(_field_cell_count * max_groups)
	buf_disp_vx = rd.storage_buffer_create(_field_cell_count * max_groups * 4, zfg)
	buf_disp_vy = rd.storage_buffer_create(_field_cell_count * max_groups * 4, zfg)


func build_uniform_sets() -> void:
	us_clear = rd.uniform_set_create([_u(0, buf_cell_count)], shd_clear, 0)

	us_prefix = rd.uniform_set_create([
		_u(0, buf_cell_count), _u(1, buf_cell_start), _u(2, buf_cell_offset),
	], shd_prefix, 0)

	us_count   = [RID(), RID()]
	us_scatter = [RID(), RID()]
	us_density = [RID(), RID()]
	us_steer   = [RID(), RID()]
	us_combat  = [RID(), RID()]

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
			_u(6, buf_agent_info), _u(7, buf_faction_presence),
			_u(8, buf_cell_attacker),
		], shd_density, 0)

		us_combat[b] = rd.uniform_set_create([
			_u(0,  buf_pos_x[b]),         _u(1,  buf_pos_y[b]),
			_u(2,  buf_cell_start),       _u(3,  buf_cell_count),
			_u(4,  buf_sorted_idx),       _u(5,  buf_agent_info),
			_u(6,  buf_damage_acc),       _u(7,  buf_cooldown),
			_u(8,  buf_atk_range),        _u(9,  buf_atk_damage),
			_u(10, buf_max_hp),           _u(11, buf_regen_rate),
			_u(12, buf_faction_presence), _u(13, _buf_goal_dist_all),
			_u(14, buf_alliance),         _u(15, buf_fac_to_group),
		], shd_combat, 0)

		var w := 1 - b
		us_steer[b] = rd.uniform_set_create([
			_u(0,  buf_pos_x[b]),  _u(1,  buf_pos_y[b]),
			_u(2,  buf_vel_x[b]),  _u(3,  buf_vel_y[b]),
			_u(4,  buf_pos_x[w]),  _u(5,  buf_pos_y[w]),
			_u(6,  buf_vel_x[w]),  _u(7,  buf_vel_y[w]),
			_u(8,  buf_cell_start), _u(9,  buf_cell_count),
			_u(10, buf_sorted_idx),
			_u(11, _buf_terrain),
			_u(12, buf_stall),
		], shd_steer, 0)

	# Set 1 for steer (not double-buffered)
	us_steer_s1 = rd.uniform_set_create([
		_u(0, buf_agent_info),
		_u(1, _buf_out_vx),
		_u(2, _buf_out_vy),
		_u(3, buf_fac_to_group),
		_u(4, buf_corpse_map),
		_u(5, buf_cell_attacker),
		_u(6, buf_cooldown),
		_u(7, buf_cell_blocked),
		_u(8, buf_disp_vx),
		_u(9, buf_disp_vy),
	], shd_steer, 1)

	# cell_blocked pass
	us_cell_blocked = rd.uniform_set_create([
		_u(0, buf_cell_attacker),
		_u(1, buf_agent_info),
		_u(2, buf_fac_to_group),
		_u(3, _buf_terrain),
		_u(4, buf_cell_blocked),
	], shd_cell_blocked, 0)

	# disp_flow pass
	us_disp_flow = rd.uniform_set_create([
		_u(0, buf_cell_blocked),
		_u(1, _buf_out_vx),
		_u(2, _buf_out_vy),
		_u(3, buf_disp_vx),
		_u(4, buf_disp_vy),
		_u(5, buf_cell_attacker),
		_u(6, buf_faction_presence),
		_u(7, buf_fac_to_group),
	], shd_disp_flow, 0)


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


func upload_combat_data(info: PackedInt32Array, max_hp_arr: PackedInt32Array,
						regen: PackedFloat32Array, atk_range_arr: PackedFloat32Array,
						atk_dmg: PackedFloat32Array, alliance_arr: PackedInt32Array,
						f2g: PackedInt32Array) -> void:
	_upi(buf_agent_info, info)
	_upi(buf_max_hp, max_hp_arr)
	_upf(buf_regen_rate, regen)
	_upf(buf_atk_range, atk_range_arr)
	_upf(buf_atk_damage, atk_dmg)
	_upi(buf_alliance, alliance_arr)
	_upi(buf_fac_to_group, f2g)
	# Reset transient combat state
	var zf := _zero_bytes(max_agents)
	rd.buffer_update(buf_damage_acc, 0, zf.size(), zf)
	rd.buffer_update(buf_cooldown, 0, zf.size(), zf)


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


func readback_agent_info() -> PackedInt32Array:
	return rd.buffer_get_data(buf_agent_info).to_int32_array()


func readback_damage_acc() -> PackedInt32Array:
	return rd.buffer_get_data(buf_damage_acc).to_int32_array()


func readback_cell_attacker() -> PackedInt32Array:
	return rd.buffer_get_data(buf_cell_attacker).to_int32_array()


func readback_cell_blocked() -> PackedInt32Array:
	return rd.buffer_get_data(buf_cell_blocked).to_int32_array()


func clear_corpse_map() -> void:
	if not buf_corpse_map.is_valid():
		return
	var ff := PackedByteArray()
	ff.resize(_field_cell_count * 4)
	ff.fill(0xFF)
	rd.buffer_update(buf_corpse_map, 0, ff.size(), ff)
	if buf_cell_attacker.is_valid():
		rd.buffer_update(buf_cell_attacker, 0, ff.size(), ff)


# ── Dispatches ───────────────────────────────────────────────────────────

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
	rd.compute_list_dispatch(cl, _live_groups(), 1, 1)
	rd.compute_list_add_barrier(cl)

	rd.compute_list_bind_compute_pipeline(cl, pip_prefix)
	rd.compute_list_bind_uniform_set(cl, us_prefix, 0)
	rd.compute_list_set_push_constant(cl, pc16, 16)
	rd.compute_list_dispatch(cl, 1, 1, 1)
	rd.compute_list_add_barrier(cl)

	rd.compute_list_bind_compute_pipeline(cl, pip_scatter)
	rd.compute_list_bind_uniform_set(cl, us_scatter[cur], 0)
	rd.compute_list_set_push_constant(cl, pc_grid, 16)
	rd.compute_list_dispatch(cl, _live_groups(), 1, 1)


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
	rd.compute_list_dispatch(cl, ceili(float(_field_gw) / 8.0), ceili(float(_field_gh) / 8.0), 1)


func dispatch_combat(cl: int, dt: float, engage_range: float,
					  attack_cd_base: float) -> void:
	var p := PackedByteArray(); p.resize(48)
	p.encode_s32(0,   agent_count)
	p.encode_s32(4,   sg_w)
	p.encode_s32(8,   sg_h)
	p.encode_float(12, sg_inv_cs)
	p.encode_float(16, dt)
	p.encode_float(20, engage_range)
	p.encode_float(24, attack_cd_base)
	p.encode_float(28, 1.0 / _field_cs)
	p.encode_s32(32,  _field_gw)
	p.encode_s32(36,  _field_gh)
	p.encode_s32(40,  _field_cell_count)
	p.encode_u32(44,  0)
	rd.compute_list_bind_compute_pipeline(cl, pip_combat)
	rd.compute_list_bind_uniform_set(cl, us_combat[cur], 0)
	rd.compute_list_set_push_constant(cl, p, 48)
	rd.compute_list_dispatch(cl, _live_groups(), 1, 1)


func dispatch_cell_blocked(cl: int) -> void:
	var p := PackedByteArray(); p.resize(16)
	p.encode_s32(0, _field_gw)
	p.encode_s32(4, _field_gh)
	p.encode_s32(8, agent_count)
	p.encode_s32(12, 0)
	rd.compute_list_bind_compute_pipeline(cl, pip_cell_blocked)
	rd.compute_list_bind_uniform_set(cl, us_cell_blocked, 0)
	rd.compute_list_set_push_constant(cl, p, 16)
	rd.compute_list_dispatch(cl, ceili(float(_field_gw) / 8.0), ceili(float(_field_gh) / 8.0), 1)


func dispatch_disp_flow(cl: int, num_groups: int, num_factions: int) -> void:
	var p := PackedByteArray(); p.resize(16)
	p.encode_s32(0, _field_gw)
	p.encode_s32(4, _field_gh)
	p.encode_s32(8, num_groups)
	p.encode_s32(12, num_factions)
	rd.compute_list_bind_compute_pipeline(cl, pip_disp_flow)
	rd.compute_list_bind_uniform_set(cl, us_disp_flow, 0)
	rd.compute_list_set_push_constant(cl, p, 16)
	rd.compute_list_dispatch(cl, ceili(float(_field_gw) / 8.0), ceili(float(_field_gh) / 8.0), num_groups)


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
	rd.compute_list_bind_uniform_set(cl, us_steer_s1, 1)
	rd.compute_list_set_push_constant(cl, p, 64)
	rd.compute_list_dispatch(cl, _live_groups(), 1, 1)


# ── Helpers ──────────────────────────────────────────────────────────────

func _live_groups() -> int:
	return maxi(ceili(float(agent_count) / 64.0), 1)

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


func _upi(buf: RID, arr: PackedInt32Array) -> void:
	var b := arr.to_byte_array()
	rd.buffer_update(buf, 0, b.size(), b)


func _zero_bytes(count: int) -> PackedByteArray:
	var a := PackedFloat32Array(); a.resize(count)
	return a.to_byte_array()


# ── Cleanup ──────────────────────────────────────────────────────────────

func cleanup() -> void:
	if rd == null:
		return
	var _free := func(rid: RID) -> void:
		if rid.is_valid():
			rd.free_rid(rid)
	for rid in [us_clear, us_prefix, us_steer_s1, us_cell_blocked, us_disp_flow]:
		_free.call(rid)
	for arr in [us_count, us_scatter, us_density, us_steer, us_combat]:
		for rid in arr:
			_free.call(rid)
	for pip in [pip_clear, pip_count, pip_prefix, pip_scatter,
				pip_density, pip_steer, pip_combat, pip_cell_blocked, pip_disp_flow]:
		_free.call(pip)
	for shd in [shd_clear, shd_count, shd_prefix, shd_scatter,
				shd_density, shd_steer, shd_combat, shd_cell_blocked, shd_disp_flow]:
		_free.call(shd)
	for arr in [buf_pos_x, buf_pos_y, buf_vel_x, buf_vel_y]:
		for rid in arr:
			_free.call(rid)
	for rid in [buf_cell_count, buf_cell_start, buf_cell_offset, buf_sorted_idx,
				buf_stall, buf_agent_info, buf_damage_acc, buf_max_hp,
				buf_regen_rate, buf_cooldown, buf_atk_range, buf_atk_damage,
				buf_faction_presence, buf_corpse_map, buf_cell_attacker,
				buf_cell_blocked, buf_disp_vx, buf_disp_vy,
				buf_alliance, buf_fac_to_group]:
		_free.call(rid)
	rd = null
