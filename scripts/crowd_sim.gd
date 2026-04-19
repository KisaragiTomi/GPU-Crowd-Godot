extends Node2D

const MAX_AGENTS := 10000
const NUM_FACTIONS := 32
const NUM_GOAL_GROUPS := 32
const DEBUG := false

@export var grid_width := 200
@export var grid_height := 120
@export var cell_size := 8.0
@export var agent_count := 2000
@export var agent_radius := 3.0
@export var separation_radius := 16.0
@export var separation_strength := 120.0
@export var steering_responsiveness := 6.0
@export var engage_range := 200.0
@export var attack_cd_base := 1.0

enum Equip { SWORD, SPEAR, BOW, CROSSBOW, SHIELD }

const EQUIP_DB := {
	Equip.SWORD:    { "melee": true,  "range": 20.0,  "damage": 15.0, "cooldown": 0.8 },
	Equip.SPEAR:    { "melee": true,  "range": 30.0,  "damage": 12.0, "cooldown": 1.0 },
	Equip.BOW:      { "melee": false, "range": 100.0, "damage": 8.0,  "cooldown": 1.2 },
	Equip.CROSSBOW: { "melee": false, "range": 70.0,  "damage": 18.0, "cooldown": 2.0 },
	Equip.SHIELD:   { "melee": false, "range": 0.0,   "damage": 0.0,  "cooldown": 0.0 },
}

var rd: RenderingDevice
var field: GPUSWEField
var agents: GPUAgents

var agent_pos: PackedVector2Array
var agent_vel: PackedVector2Array
var display_pos: PackedVector2Array
var _pos_buf: Array[PackedVector2Array] = []
var _pos_buf_idx := 0
const POS_BUF_FRAMES := 3

var world_size: Vector2
var wall_cells: PackedVector2Array
var goal_cells: PackedVector2Array

var swe_accum := 0.0
var swe_interval := 0.033

var show_density := false
var show_velocity := false
var show_goal := false
var paused := false

# Brush painting
enum BrushMode { NONE, WALL, GOAL, ERASE }
var brush_mode: BrushMode = BrushMode.NONE
var brush_radius := 2
var is_painting := false
var paint_erase := false
var goal_mask: PackedByteArray
var terrain_dirty := false
var goal_dirty := false

var perf_gpu := 0.0
var perf_read := 0.0
var perf_disppos := 0.0
var perf_mesh := 0.0
var perf_goal := 0.0
var perf_alpha := 0.15
var perf_peak_total := 0.0
var perf_peak_detail := ""
var alive_count := 0
var selected_agent: int = -1
var nearest_enemy: int = -1
var select_label: Label

var multi_mesh: MultiMesh
var mm_instance: MultiMeshInstance2D
var mm_buf := PackedFloat32Array()

# Per-agent CPU data
var agent_equips: Array
var agent_factions: PackedInt32Array
var cpu_agent_info: PackedInt32Array
var cpu_max_hp: PackedInt32Array
var cpu_regen_rate: PackedFloat32Array
var cpu_atk_range: PackedFloat32Array
var cpu_atk_damage: PackedFloat32Array

# Faction config
var alliance_masks: PackedInt32Array
var faction_to_group: PackedInt32Array
var faction_colors: Array[Color]
var group_goals: Array
var goal_update_timer := 0.0
var goal_update_interval := 0.5
var _goal_rr_idx := 0
var cached_info := PackedInt32Array()
var cached_damage := PackedInt32Array()
var cached_cell_atk := PackedInt32Array()
var cached_cell_blocked := PackedInt32Array()

#region debug — jitter / crossing / DISP monitor variables
var _dbg_prev_pos: Dictionary = {}
var _dbg_jitter_frames: Dictionary = {}
var _dbg_jitter_log: Array = []
var _dbg_jitter_agents: PackedInt32Array = PackedInt32Array()
var _dbg_recording := false
var _dbg_pre_buffer: Array = []
const _DBG_MAX_FRAMES := 60
const _DBG_PRE_FRAMES := 10
const _DBG_JITTER_SPD := 65.0
const _DBG_JITTER_TRIGGER := 3
var _dbg_jitter_marked: Dictionary = {}
var _dbg_cross_timer := 0
const _DBG_CROSS_INTERVAL := 30
const _DBG_CROSS_DIST := 16.0
var _dbg_disp_frames: Dictionary = {}
const _DBG_DISP_TRIGGER := 120
var _dbg_disp_recording := false
var _dbg_auto_pause_done := true
var _dbg_disp_log: Array = []
var _dbg_disp_agent := -1
const _DBG_DISP_LOG_FRAMES := 120
#endregion

# HUD
var hud_label: Label
var slider_agents: HSlider
var label_agents: Label
var slider_speed: HSlider
var slider_density_scale: HSlider
var slider_separation: HSlider
var slider_engage: HSlider
var slider_cooldown: HSlider
var btn_density: CheckButton
var btn_velocity: CheckButton
var btn_goal: CheckButton
var btn_pause: Button
var cam: Camera2D
var cam_zoom := 1.0
var cam_dragging := false
var cam_drag_origin := Vector2.ZERO


func _ready() -> void:
	world_size = Vector2(grid_width * cell_size, grid_height * cell_size)

	cam = $Camera2D

	rd = RenderingServer.create_local_rendering_device()
	assert(rd != null, "Vulkan RenderingDevice required")

	_setup_factions()

	field = GPUSWEField.new(rd, grid_width, grid_height, cell_size, NUM_GOAL_GROUPS)
	field.goal_scale = 0.0
	_build_environment()
	field.upload_static_data()

	agents = GPUAgents.new(rd, MAX_AGENTS, separation_radius, world_size)
	agents.set_field_buffers(
		field.buf_out_vx, field.buf_out_vy,
		field.buf_terrain, field.buf_goal_dist,
		field.buf_density,
		grid_width, grid_height, cell_size,
		field.buf_goal_dist_all, NUM_GOAL_GROUPS
	)
	agents.build_uniform_sets()
	field.setup_bfs(agents.buf_faction_presence, agents.buf_fac_to_group)

	_spawn_agents()
	agents.upload_agents(agent_pos, agent_vel, agent_count)
	_upload_combat()

	_setup_multimesh()
	_setup_hud()


func _exit_tree() -> void:
	pass


# ── Faction / Alliance setup ─────────────────────────────────────────────

func _setup_factions() -> void:
	alliance_masks = PackedInt32Array()
	alliance_masks.resize(NUM_FACTIONS)
	for f in range(NUM_FACTIONS):
		alliance_masks[f] = 1 << f

	faction_to_group = PackedInt32Array()
	faction_to_group.resize(NUM_FACTIONS)
	for f in range(NUM_FACTIONS):
		faction_to_group[f] = f

	faction_colors = []
	for f in range(NUM_FACTIONS):
		var hue := float(f) / float(NUM_FACTIONS)
		faction_colors.append(Color.from_hsv(hue, 0.85, 0.95))


# ── Environment ──────────────────────────────────────────────────────────

var castle_centers: Array[Vector2i] = []
var castle_exits: Array[Vector2i] = []

func _build_environment() -> void:
	field.add_wall(0, 0, grid_width, 1)
	field.add_wall(0, grid_height - 1, grid_width, grid_height)
	field.add_wall(0, 0, 1, grid_height)
	field.add_wall(grid_width - 1, 0, grid_width, grid_height)

	_gen_castles()

	_update_wall_cells()
	_setup_goals()


func _gen_castles() -> void:
	const FOOT := 10
	const EXIT_W := 2

	castle_centers.clear()
	castle_exits.clear()

	var mcx := grid_width / 2
	var mcy := grid_height / 2
	var occupied: Array[Rect2i] = []

	for _attempt in range(2000):
		if castle_centers.size() >= NUM_FACTIONS:
			break
		var rx := randi_range(3, grid_width - FOOT - 3)
		var ry := randi_range(3, grid_height - FOOT - 3)
		var guard := Rect2i(rx - 2, ry - 2, FOOT + 4, FOOT + 4)
		var overlap := false
		for o in occupied:
			if guard.intersects(o):
				overlap = true
				break
		if overlap:
			continue
		occupied.append(Rect2i(rx, ry, FOOT, FOOT))

		var dx := float(mcx - (rx + FOOT / 2))
		var dy := float(mcy - (ry + FOOT / 2))
		var exit_side := 0
		if absf(dx) > absf(dy):
			exit_side = 3 if dx > 0 else 1
		else:
			exit_side = 0 if dy > 0 else 2

		var x0 := rx; var y0 := ry
		var x1 := rx + FOOT; var y1 := ry + FOOT
		var em_h := rx + FOOT / 2 - EXIT_W / 2
		var em_v := ry + FOOT / 2 - EXIT_W / 2

		if exit_side == 0:
			field.add_wall(x0, y0, em_h, y0 + 1)
			field.add_wall(em_h + EXIT_W, y0, x1, y0 + 1)
		else:
			field.add_wall(x0, y0, x1, y0 + 1)

		if exit_side == 2:
			field.add_wall(x0, y1 - 1, em_h, y1)
			field.add_wall(em_h + EXIT_W, y1 - 1, x1, y1)
		else:
			field.add_wall(x0, y1 - 1, x1, y1)

		if exit_side == 3:
			field.add_wall(x0, y0 + 1, x0 + 1, em_v)
			field.add_wall(x0, em_v + EXIT_W, x0 + 1, y1 - 1)
		else:
			field.add_wall(x0, y0 + 1, x0 + 1, y1 - 1)

		if exit_side == 1:
			field.add_wall(x1 - 1, y0 + 1, x1, em_v)
			field.add_wall(x1 - 1, em_v + EXIT_W, x1, y1 - 1)
		else:
			field.add_wall(x1 - 1, y0 + 1, x1, y1 - 1)

		var ex := 0; var ey := 0
		match exit_side:
			0: ex = em_h; ey = y0
			1: ex = x1 - 1; ey = em_v
			2: ex = em_h; ey = y1 - 1
			3: ex = x0; ey = em_v

		castle_centers.append(Vector2i(rx + FOOT / 2, ry + FOOT / 2))
		castle_exits.append(Vector2i(ex, ey))


func _setup_goals() -> void:
	goal_mask = PackedByteArray()
	goal_mask.resize(grid_width * grid_height)
	goal_cells = PackedVector2Array()

	var center_goal: Array[Vector2i] = [Vector2i(grid_width / 2, grid_height / 2)]
	group_goals = []
	for _g in range(NUM_GOAL_GROUPS):
		group_goals.append(center_goal)
	field.build_all_goal_fields(group_goals)


func _update_dynamic_goals_cpu() -> void:
	if cached_info.is_empty():
		return
	var half := NUM_FACTIONS / 2
	var used_a := PackedByteArray()
	used_a.resize(grid_width * grid_height)
	var used_b := PackedByteArray()
	used_b.resize(grid_width * grid_height)
	var team_a_cells: Array[Vector2i] = []
	var team_b_cells: Array[Vector2i] = []
	for a_idx in range(agent_count):
		if (cached_info[a_idx] & 1) == 0:
			continue
		var fac := (cached_info[a_idx] >> 1) & 0x1F
		var cx := clampi(int(agent_pos[a_idx].x / cell_size), 1, grid_width - 2)
		var cy := clampi(int(agent_pos[a_idx].y / cell_size), 1, grid_height - 2)
		var ci := cy * grid_width + cx
		if fac < half:
			if used_a[ci] == 0:
				used_a[ci] = 1
				team_a_cells.append(Vector2i(cx, cy))
		else:
			if used_b[ci] == 0:
				used_b[ci] = 1
				team_b_cells.append(Vector2i(cx, cy))
	if team_b_cells.is_empty() or team_a_cells.is_empty():
		return
	const MAX_SEEDS := 256
	if team_a_cells.size() > MAX_SEEDS:
		team_a_cells = _subsample(team_a_cells, MAX_SEEDS)
	if team_b_cells.size() > MAX_SEEDS:
		team_b_cells = _subsample(team_b_cells, MAX_SEEDS)
	group_goals = [team_b_cells, team_a_cells]
	field.build_all_goal_fields(group_goals)
	field.upload_goal_data()


func _subsample(arr: Array[Vector2i], n: int) -> Array[Vector2i]:
	var out: Array[Vector2i] = []
	var step := float(arr.size()) / float(n)
	for i in range(n):
		out.append(arr[int(i * step)])
	return out


# ── Agents ───────────────────────────────────────────────────────────────

func _spawn_agents() -> void:
	agent_pos = PackedVector2Array(); agent_pos.resize(agent_count)
	agent_vel = PackedVector2Array(); agent_vel.resize(agent_count)
	display_pos = PackedVector2Array(); display_pos.resize(agent_count)
	_pos_buf.clear()
	for _f in range(POS_BUF_FRAMES):
		var buf := PackedVector2Array(); buf.resize(agent_count)
		_pos_buf.append(buf)
	_pos_buf_idx = 0
	agent_equips = []
	agent_factions = PackedInt32Array(); agent_factions.resize(agent_count)

	cpu_agent_info = PackedInt32Array(); cpu_agent_info.resize(MAX_AGENTS)
	cpu_max_hp     = PackedInt32Array(); cpu_max_hp.resize(MAX_AGENTS)
	cpu_regen_rate = PackedFloat32Array(); cpu_regen_rate.resize(MAX_AGENTS)
	cpu_atk_range  = PackedFloat32Array(); cpu_atk_range.resize(MAX_AGENTS)
	cpu_atk_damage = PackedFloat32Array(); cpu_atk_damage.resize(MAX_AGENTS)

	for i in range(agent_count):
		var faction := i % NUM_FACTIONS
		agent_factions[i] = faction
		if faction < castle_centers.size():
			var cc := castle_centers[faction]
			agent_pos[i] = Vector2(
				(float(cc.x) + randf_range(-3.0, 3.0)) * cell_size,
				(float(cc.y) + randf_range(-3.0, 3.0)) * cell_size)
		else:
			agent_pos[i] = Vector2(
				randf_range(cell_size * 3.0, world_size.x - cell_size * 3.0),
				randf_range(cell_size * 3.0, world_size.y - cell_size * 3.0))
		agent_vel[i] = Vector2.ZERO
		display_pos[i] = agent_pos[i]
		for f in range(POS_BUF_FRAMES):
			_pos_buf[f][i] = agent_pos[i]

		# Equipment
		var equips: Array = []
		if randf() < 0.6:
			equips.append(Equip.SWORD if randf() < 0.5 else Equip.SPEAR)
		if randf() < 0.4:
			equips.append(Equip.BOW if randf() < 0.5 else Equip.CROSSBOW)
		if equips.is_empty():
			equips.append(Equip.SWORD)
		if randf() < 0.3:
			equips.append(Equip.SHIELD)
		agent_equips.append(equips)

		var has_melee := false
		var has_ranged := false
		var has_shield := false
		var best_range := 0.0
		var best_dmg := 0.0
		for e in equips:
			var db: Dictionary = EQUIP_DB[e]
			if db.melee:
				has_melee = true
			elif db.range > 0:
				has_ranged = true
			if e == Equip.SHIELD:
				has_shield = true
			best_range = maxf(best_range, db.range)
			best_dmg = maxf(best_dmg, db.damage)

		var info: int = 1
		info |= (faction & 0x1F) << 1
		if has_melee:  info |= (1 << 6)
		if has_ranged: info |= (1 << 7)
		if has_shield: info |= (1 << 8)

		cpu_agent_info[i] = info
		cpu_max_hp[i] = 20000
		cpu_regen_rate[i] = 2.0
		cpu_atk_range[i] = best_range
		cpu_atk_damage[i] = best_dmg


func _upload_combat() -> void:
	agents.upload_combat_data(
		cpu_agent_info, cpu_max_hp,
		cpu_regen_rate, cpu_atk_range, cpu_atk_damage,
		alliance_masks, faction_to_group
	)


# ── Brush painting ───────────────────────────────────────────────────────

func _update_wall_cells() -> void:
	wall_cells = PackedVector2Array()
	for y in range(grid_height):
		for x in range(grid_width):
			if field.terrain[field.idx(x, y)] > 0.5:
				wall_cells.append(Vector2(x, y))


func _rebuild_goals() -> void:
	var goals_0: Array[Vector2i] = []
	goal_cells = PackedVector2Array()
	for y in range(grid_height):
		for x in range(grid_width):
			if goal_mask[y * grid_width + x] > 0:
				goals_0.append(Vector2i(x, y))
				goal_cells.append(Vector2(x, y))
	if goals_0.is_empty():
		goals_0.append(Vector2i(grid_width - 2, grid_height / 2))
		goal_cells.append(Vector2(grid_width - 2, grid_height / 2))
	if group_goals.size() > 1:
		for g in group_goals[1]:
			goal_cells.append(Vector2(g.x, g.y))
	group_goals[0] = goals_0
	field.build_all_goal_fields(group_goals)


func _apply_brush(world_pos: Vector2) -> void:
	var cx := int(world_pos.x / cell_size)
	var cy := int(world_pos.y / cell_size)
	var r := brush_radius
	var changed := false

	for dy in range(-r, r + 1):
		for dx in range(-r, r + 1):
			if dx * dx + dy * dy > r * r:
				continue
			var gx := cx + dx
			var gy := cy + dy
			if gx < 1 or gx >= grid_width - 1 or gy < 1 or gy >= grid_height - 1:
				continue
			var gi := field.idx(gx, gy)

			if brush_mode == BrushMode.WALL:
				var want := 0.0 if paint_erase else 1.0
				if field.terrain[gi] != want:
					field.terrain[gi] = want
					if paint_erase:
						goal_mask[gy * grid_width + gx] = 0
					changed = true
					terrain_dirty = true
					goal_dirty = true

			elif brush_mode == BrushMode.GOAL:
				var want: int = 0 if paint_erase else 1
				if goal_mask[gy * grid_width + gx] != want and field.terrain[gi] < 0.5:
					goal_mask[gy * grid_width + gx] = want
					changed = true
					goal_dirty = true

			elif brush_mode == BrushMode.ERASE:
				if field.terrain[gi] > 0.5:
					field.terrain[gi] = 0.0
					changed = true
					terrain_dirty = true
					goal_dirty = true
				if goal_mask[gy * grid_width + gx] > 0:
					goal_mask[gy * grid_width + gx] = 0
					changed = true
					goal_dirty = true

	if changed:
		queue_redraw()


func _flush_paint() -> void:
	if terrain_dirty:
		_update_wall_cells()
		field._upload(field.buf_terrain, field.terrain)
		terrain_dirty = false
	if goal_dirty:
		_rebuild_goals()
		field.upload_static_data()
		goal_dirty = false


# ── MultiMesh ────────────────────────────────────────────────────────────

func _setup_multimesh() -> void:
	var quad := QuadMesh.new()
	quad.size = Vector2(agent_radius * 2.0, agent_radius * 2.0)
	multi_mesh = MultiMesh.new()
	multi_mesh.transform_format = MultiMesh.TRANSFORM_2D
	multi_mesh.use_colors = true
	multi_mesh.instance_count = MAX_AGENTS
	multi_mesh.visible_instance_count = agent_count
	multi_mesh.mesh = quad
	mm_instance = MultiMeshInstance2D.new()
	mm_instance.multimesh = multi_mesh
	mm_instance.texture = _make_circle_texture(12)
	add_child(mm_instance)


func _make_circle_texture(radius_px: int) -> ImageTexture:
	var size := radius_px * 2
	var img := Image.create(size, size, false, Image.FORMAT_RGBA8)
	var center := Vector2(float(radius_px), float(radius_px))
	for y in range(size):
		for x in range(size):
			if Vector2(float(x) + 0.5, float(y) + 0.5).distance_to(center) <= float(radius_px):
				img.set_pixel(x, y, Color.WHITE)
			else:
				img.set_pixel(x, y, Color(1, 1, 1, 0))
	return ImageTexture.create_from_image(img)


func _update_display_pos(_dt: float) -> void:
	var cur := _pos_buf[_pos_buf_idx]
	for i in range(agent_count):
		cur[i] = agent_pos[i]
	_pos_buf_idx = (_pos_buf_idx + 1) % POS_BUF_FRAMES
	var inv := 1.0 / float(POS_BUF_FRAMES)
	for i in range(agent_count):
		var avg := Vector2.ZERO
		for f in range(POS_BUF_FRAMES):
			avg += _pos_buf[f][i]
		display_pos[i] = avg * inv


func _sync_multimesh() -> void:
	cached_info = agents.readback_agent_info()
	cached_damage = agents.readback_damage_acc()
	if selected_agent >= 0:
		cached_cell_atk = agents.readback_cell_attacker()
		cached_cell_blocked = agents.readback_cell_blocked()
	alive_count = 0
	var stride := 12
	if mm_buf.size() != MAX_AGENTS * stride:
		mm_buf.resize(MAX_AGENTS * stride)
	for i in range(agent_count):
		var off := i * stride
		var ainfo := cached_info[i]
		if (ainfo & 1) != 0:
			var fac := (ainfo >> 1) & 0x1F
			var hp_frac := 1.0 - clampf(float(cached_damage[i]) / maxf(float(cpu_max_hp[i]), 1.0), 0.0, 1.0)
			var brightness := lerpf(0.2, 1.0, hp_frac)
			var base_col := faction_colors[fac]
			var pos := display_pos[i]
			mm_buf[off]     = 1.0
			mm_buf[off + 1] = 0.0
			mm_buf[off + 2] = 0.0
			mm_buf[off + 3] = pos.x
			mm_buf[off + 4] = 0.0
			mm_buf[off + 5] = 1.0
			mm_buf[off + 6] = 0.0
			mm_buf[off + 7] = pos.y
			if DEBUG and _dbg_jitter_marked.has(i):
				mm_buf[off + 8]  = 1.0
				mm_buf[off + 9]  = 1.0
				mm_buf[off + 10] = 1.0
				mm_buf[off + 11] = 1.0
			else:
				mm_buf[off + 8]  = base_col.r * brightness
				mm_buf[off + 9]  = base_col.g * brightness
				mm_buf[off + 10] = base_col.b * brightness
				mm_buf[off + 11] = 1.0
			alive_count += 1
		else:
			mm_buf[off]     = 1.0
			mm_buf[off + 1] = 0.0
			mm_buf[off + 2] = 0.0
			mm_buf[off + 3] = -10000.0
			mm_buf[off + 4] = 0.0
			mm_buf[off + 5] = 1.0
			mm_buf[off + 6] = 0.0
			mm_buf[off + 7] = -10000.0
			mm_buf[off + 8]  = 0.0
			mm_buf[off + 9]  = 0.0
			mm_buf[off + 10] = 0.0
			mm_buf[off + 11] = 0.0
	multi_mesh.set_buffer(mm_buf)

	if DEBUG and not _dbg_auto_pause_done:
		for j in range(agent_count):
			if j >= cached_info.size(): break
			var aj := cached_info[j]
			if (aj & 1) != 0 and (aj & (1 << 10)) != 0:
				selected_agent = j
				_dbg_auto_pause_done = true
				print("=== AUTO SELECT: DISP Agent #%d ===" % j)
				paused = true
				queue_redraw()
				break


func _pick_agent(world_pos: Vector2) -> void:
	var best := -1
	var best_dsq := agent_radius * agent_radius * 9.0
	for idx in range(agent_count):
		if cached_info.size() > idx and (cached_info[idx] & 1) == 0:
			continue
		var dsq := agent_pos[idx].distance_squared_to(world_pos)
		if dsq < best_dsq:
			best_dsq = dsq
			best = idx
	selected_agent = best
	_update_select_label()
	queue_redraw()


func _update_select_label() -> void:
	if select_label == null:
		return
	if selected_agent < 0 or selected_agent >= agent_count:
		select_label.text = ""
		return
	var idx := selected_agent
	var alive := cached_info.size() > idx and (cached_info[idx] & 1) != 0
	var fac := 0
	var is_atk := false
	var is_disp := false
	if cached_info.size() > idx:
		fac = (cached_info[idx] >> 1) & 0x1F
		is_atk = (cached_info[idx] & (1 << 9)) != 0
		is_disp = (cached_info[idx] & (1 << 10)) != 0
	var team := "G%d" % fac
	var dmg := 0
	if cached_damage.size() > idx:
		dmg = cached_damage[idx]
	var max_hp_val := cpu_max_hp[idx] if idx < cpu_max_hp.size() else 20000
	var hp_cur := maxf(float(max_hp_val - dmg) / 100.0, 0.0)
	var hp_max := float(max_hp_val) / 100.0

	var equip_names: PackedStringArray = PackedStringArray()
	if idx < agent_equips.size():
		for e in agent_equips[idx]:
			match e:
				Equip.SWORD:    equip_names.append("Sword")
				Equip.SPEAR:    equip_names.append("Spear")
				Equip.BOW:      equip_names.append("Bow")
				Equip.CROSSBOW: equip_names.append("Crossbow")
				Equip.SHIELD:   equip_names.append("Shield")
	var equip_str := ", ".join(equip_names)

	var rng_val := cpu_atk_range[idx] if idx < cpu_atk_range.size() else 0.0
	var dmg_val := cpu_atk_damage[idx] if idx < cpu_atk_damage.size() else 0.0
	var status := "DEAD" if not alive else ("Attacking" if is_atk else "Idle")
	var pos_v := agent_pos[idx] if idx < agent_pos.size() else Vector2.ZERO

	# Find nearest enemy
	nearest_enemy = -1
	var my_ally_mask := alliance_masks[fac] if fac < alliance_masks.size() else 0
	var ne_dsq := INF
	for j in range(agent_count):
		if j == idx:
			continue
		if cached_info.size() <= j or (cached_info[j] & 1) == 0:
			continue
		var jfac := (cached_info[j] >> 1) & 0x1F
		if (my_ally_mask >> jfac) & 1:
			continue
		var dsq := agent_pos[idx].distance_squared_to(agent_pos[j])
		if dsq < ne_dsq:
			ne_dsq = dsq
			nearest_enemy = j
	var ne_str := ""
	if nearest_enemy >= 0:
		var ne_dist := sqrt(ne_dsq)
		var ne_fac := (cached_info[nearest_enemy] >> 1) & 0x1F
		ne_str = "Nearest Enemy: #%d (Fac %d) dist=%.1f%s" % [
			nearest_enemy, ne_fac, ne_dist,
			"  IN RANGE" if ne_dist <= rng_val else ""]

	select_label.text = (
		"Agent #%d  [Faction %d / Team %s]\n" % [idx, fac, team]
		+ "HP: %.1f / %.1f\n" % [hp_cur, hp_max]
		+ "Equip: %s\n" % equip_str
		+ "Range: %.0f  Dmg: %.0f\n" % [rng_val, dmg_val]
		+ "Status: %s\n" % status
		+ "Pos: (%.1f, %.1f)\n" % [pos_v.x, pos_v.y]
		+ ne_str
	)


# ── Simulation loop ─────────────────────────────────────────────────────

func _physics_process(dt: float) -> void:
	_flush_paint()
	if paused:
		return

	var t0 := Time.get_ticks_usec()

	var cl := rd.compute_list_begin()

	# 1 — Build cell buffer
	agents.dispatch_build_grid(cl)
	rd.compute_list_add_barrier(cl)

	# 2 — Density + faction presence bitmap
	agents.dispatch_density(cl)
	rd.compute_list_add_barrier(cl)

	# 3 — SWE flux (throttled)
	swe_accum += dt
	if swe_accum >= swe_interval:
		field.dispatch_swe(cl, swe_accum)
		swe_accum = 0.0
		rd.compute_list_add_barrier(cl)

	# 4 — Velocity field (multi-group)
	field.dispatch_velocity(cl, NUM_GOAL_GROUPS)
	rd.compute_list_add_barrier(cl)

	# 5 — Combat
	agents.dispatch_combat(cl, dt, engage_range, attack_cd_base)
	rd.compute_list_add_barrier(cl)

	# 5.5 — Cell blocked map (per-group blockage bitmask)
	agents.dispatch_cell_blocked(cl)
	rd.compute_list_add_barrier(cl)

	# 5.6 — DISP flow field (escape direction for displaced agents)
	agents.dispatch_disp_flow(cl, NUM_GOAL_GROUPS, NUM_FACTIONS)
	rd.compute_list_add_barrier(cl)

	# 6 — Agent steer + integrate
	agents.dispatch_steer(cl, dt, separation_radius, separation_strength,
						  steering_responsiveness, world_size.x, world_size.y)

	rd.compute_list_end()
	rd.submit()
	rd.sync()
	var t1 := Time.get_ticks_usec()

	# 7 — Readback
	agent_pos = agents.readback_positions()
	var t1b := Time.get_ticks_usec()
	_update_display_pos(dt)
	var t2 := Time.get_ticks_usec()

	# 8 — MultiMesh
	_sync_multimesh()
	var t2b := Time.get_ticks_usec()
	_update_select_label()
	if DEBUG:
		_log_bow_frame(dt)

	# 9 — Dynamic goal update (GPU BFS, 1 group per frame round-robin)
	var goal_ms := 0.0
	var tg0 := Time.get_ticks_usec()
	var gcl := rd.compute_list_begin()
	field.dispatch_goal_bfs(gcl, _goal_rr_idx, NUM_FACTIONS)
	rd.compute_list_end()
	rd.submit()
	rd.sync()
	_goal_rr_idx = (_goal_rr_idx + 1) % NUM_GOAL_GROUPS
	goal_ms = float(Time.get_ticks_usec() - tg0) / 1000.0
	perf_goal = lerpf(perf_goal, goal_ms, perf_alpha)

	var t3 := Time.get_ticks_usec()

	if show_density:
		field.readback_density()
	if show_velocity:
		field.readback_velocity()

	var a := perf_alpha
	var gpu_ms := float(t1 - t0) / 1000.0
	var rd_ms  := float(t1b - t1) / 1000.0
	var dp_ms  := float(t2 - t1b) / 1000.0
	var mm_ms  := float(t2b - t2) / 1000.0
	perf_gpu     = lerpf(perf_gpu,     gpu_ms, a)
	perf_read    = lerpf(perf_read,    rd_ms, a)
	perf_disppos = lerpf(perf_disppos, dp_ms, a)
	perf_mesh    = lerpf(perf_mesh,    mm_ms, a)
	var total := perf_gpu + perf_read + perf_disppos + perf_mesh
	var frame_total := float(t3 - t0) / 1000.0
	if frame_total > perf_peak_total:
		perf_peak_total = frame_total
		perf_peak_detail = "GPU=%.1f Rd=%.1f DP=%.1f MM=%.1f G=%.1f" % [
			gpu_ms, rd_ms, dp_ms, mm_ms, goal_ms]
	if frame_total > 50.0:
		print("[SPIKE] %.1f ms | GPU=%.1f Rd=%.1f DP=%.1f MM=%.1f Goal=%.1f" % [
			frame_total, gpu_ms, rd_ms, dp_ms, mm_ms, goal_ms])
	hud_label.text = (
		"Total: %.2f ms | FPS: %d | Alive: %d / %d\n" % [total, Engine.get_frames_per_second(), alive_count, agent_count]
		+ "  GPU pipeline  %.2f ms\n" % perf_gpu
		+ "  Readback(pos) %.2f ms\n" % perf_read
		+ "  DisplayPos    %.2f ms\n" % perf_disppos
		+ "  SyncMultiMesh %.2f ms\n" % perf_mesh
		+ "  GoalUpdate    %.2f ms\n" % perf_goal
		+ "  PEAK: %.2f ms  %s" % [perf_peak_total, perf_peak_detail]
	)

	if show_density or show_velocity or show_goal or brush_mode != BrushMode.NONE or selected_agent >= 0:
		queue_redraw()


# ── HUD ──────────────────────────────────────────────────────────────────

func _setup_hud() -> void:
	var canvas := CanvasLayer.new()
	canvas.layer = 100
	add_child(canvas)

	hud_label = Label.new()
	hud_label.position = Vector2(10, 8)
	hud_label.add_theme_color_override("font_color", Color(0.85, 0.85, 0.85))
	hud_label.add_theme_font_size_override("font_size", 13)
	canvas.add_child(hud_label)

	select_label = Label.new()
	select_label.position = Vector2(10, 80)
	select_label.add_theme_color_override("font_color", Color(1.0, 0.95, 0.6))
	select_label.add_theme_font_size_override("font_size", 12)
	canvas.add_child(select_label)

	var panel := PanelContainer.new()
	panel.position = Vector2(world_size.x - 270, 40)
	panel.custom_minimum_size = Vector2(260, 0)
	var style := StyleBoxFlat.new()
	style.bg_color = Color(0.12, 0.12, 0.16, 0.92)
	style.corner_radius_top_left = 6;  style.corner_radius_top_right = 6
	style.corner_radius_bottom_left = 6; style.corner_radius_bottom_right = 6
	style.content_margin_left = 12; style.content_margin_right = 12
	style.content_margin_top = 10;  style.content_margin_bottom = 10
	panel.add_theme_stylebox_override("panel", style)
	canvas.add_child(panel)

	var vbox := VBoxContainer.new()
	vbox.add_theme_constant_override("separation", 4)
	panel.add_child(vbox)

	var title := Label.new()
	title.text = "FluidCrowd  [Combat v2]"
	title.add_theme_font_size_override("font_size", 16)
	title.add_theme_color_override("font_color", Color(0.95, 0.7, 0.3))
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(title)

	vbox.add_child(_sep())

	label_agents = Label.new()
	label_agents.add_theme_font_size_override("font_size", 12)
	vbox.add_child(label_agents)
	slider_agents = _slider(50, MAX_AGENTS, agent_count, 50)
	slider_agents.value_changed.connect(_on_agents_changed)
	vbox.add_child(slider_agents)
	_update_agent_label()

	vbox.add_child(_sep())

	vbox.add_child(_lbl("Agent Speed"))
	slider_speed = _slider(10, 200, field.agent_speed, 5)
	slider_speed.value_changed.connect(func(v: float): field.agent_speed = v)
	vbox.add_child(slider_speed)

	vbox.add_child(_lbl("Density Pressure"))
	slider_density_scale = _slider(0, 10, field.density_scale, 0.5)
	slider_density_scale.value_changed.connect(func(v: float): field.density_scale = v)
	vbox.add_child(slider_density_scale)

	vbox.add_child(_lbl("Separation Force"))
	slider_separation = _slider(0, 400, separation_strength, 10)
	slider_separation.value_changed.connect(func(v: float): separation_strength = v)
	vbox.add_child(slider_separation)

	vbox.add_child(_sep())

	vbox.add_child(_lbl("Engage Range (cells)"))
	slider_engage = _slider(5, 300, engage_range, 5)
	slider_engage.value_changed.connect(func(v: float): engage_range = v)
	vbox.add_child(slider_engage)

	vbox.add_child(_lbl("Attack Cooldown (s)"))
	slider_cooldown = _slider(0.1, 5.0, attack_cd_base, 0.1)
	slider_cooldown.value_changed.connect(func(v: float): attack_cd_base = v)
	vbox.add_child(slider_cooldown)

	vbox.add_child(_sep())

	btn_density = _toggle("Density heatmap", show_density)
	btn_density.toggled.connect(func(on: bool): show_density = on; queue_redraw())
	vbox.add_child(btn_density)
	btn_velocity = _toggle("Velocity field", show_velocity)
	btn_velocity.toggled.connect(func(on: bool): show_velocity = on; queue_redraw())
	vbox.add_child(btn_velocity)
	btn_goal = _toggle("Goal distance", show_goal)
	btn_goal.toggled.connect(func(on: bool): show_goal = on; queue_redraw())
	vbox.add_child(btn_goal)

	vbox.add_child(_sep())

	var hbox := HBoxContainer.new()
	hbox.add_theme_constant_override("separation", 8)
	vbox.add_child(hbox)
	btn_pause = Button.new()
	btn_pause.text = "  Pause  "
	btn_pause.pressed.connect(_toggle_pause)
	hbox.add_child(btn_pause)
	var btn_reset := Button.new()
	btn_reset.text = "  Reset  "
	btn_reset.pressed.connect(_reset_sim)
	hbox.add_child(btn_reset)

	vbox.add_child(_sep())
	vbox.add_child(_lbl("Brush  [1]None [2]Wall [3]Goal [4]Erase"))
	var brush_lbl := Label.new()
	brush_lbl.name = "BrushLabel"
	brush_lbl.add_theme_font_size_override("font_size", 12)
	brush_lbl.add_theme_color_override("font_color", Color(0.9, 0.75, 0.4))
	brush_lbl.text = "Mode: None | Radius: %d" % brush_radius
	vbox.add_child(brush_lbl)
	var slider_brush := _slider(1, 8, brush_radius, 1)
	slider_brush.value_changed.connect(func(v: float):
		brush_radius = int(v)
		brush_lbl.text = "Mode: %s | Radius: %d" % [_brush_name(), brush_radius]
		queue_redraw()
	)
	vbox.add_child(slider_brush)


func _lbl(txt: String) -> Label:
	var l := Label.new()
	l.text = txt
	l.add_theme_font_size_override("font_size", 12)
	l.add_theme_color_override("font_color", Color(0.7, 0.7, 0.7))
	return l

func _slider(lo: float, hi: float, val: float, step_val: float) -> HSlider:
	var s := HSlider.new()
	s.min_value = lo; s.max_value = hi; s.value = val; s.step = step_val
	s.custom_minimum_size.x = 230
	return s

func _toggle(label: String, initial: bool) -> CheckButton:
	var b := CheckButton.new()
	b.text = label; b.button_pressed = initial
	b.add_theme_font_size_override("font_size", 12)
	return b

func _sep() -> HSeparator:
	var s := HSeparator.new()
	s.add_theme_constant_override("separation", 6)
	return s

func _brush_name() -> String:
	match brush_mode:
		BrushMode.WALL: return "Wall"
		BrushMode.GOAL: return "Goal"
		BrushMode.ERASE: return "Erase"
		_: return "None"

func _update_agent_label() -> void:
	label_agents.text = "Agents: %d" % agent_count
	label_agents.add_theme_color_override("font_color", Color(0.9, 0.75, 0.4))

func _on_agents_changed(val: float) -> void:
	agent_count = int(val)
	selected_agent = -1
	nearest_enemy = -1
	_update_agent_label()
	multi_mesh.visible_instance_count = agent_count
	_spawn_agents()
	agents.upload_agents(agent_pos, agent_vel, agent_count)
	_upload_combat()
	agents.clear_corpse_map()
	_sync_multimesh()
	goal_update_timer = goal_update_interval

func _toggle_pause() -> void:
	paused = not paused
	btn_pause.text = "  Resume  " if paused else "  Pause  "
	queue_redraw()

func _reset_sim() -> void:
	selected_agent = -1
	nearest_enemy = -1
	field.reset_flux()
	_spawn_agents()
	agents.upload_agents(agent_pos, agent_vel, agent_count)
	_upload_combat()
	agents.clear_corpse_map()
	_setup_goals()
	field.upload_static_data()
	_sync_multimesh()
	goal_update_timer = 0.0


# ── Drawing ──────────────────────────────────────────────────────────────

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, world_size), Color(0.11, 0.11, 0.14))
	var cs_val := cell_size

	for wc in wall_cells:
		draw_rect(Rect2(wc.x * cs_val, wc.y * cs_val, cs_val, cs_val), Color(0.35, 0.35, 0.42))
	for gc in goal_cells:
		draw_rect(Rect2(gc.x * cs_val, gc.y * cs_val, cs_val, cs_val), Color(0.2, 0.75, 0.35, 0.18))

	if show_density:
		for y in range(grid_height):
			for x in range(grid_width):
				var d := field.density[field.idx(x, y)]
				if d > 0.05:
					draw_rect(Rect2(x * cs_val, y * cs_val, cs_val, cs_val),
							  Color(1.0, 0.15, 0.05, clampf(d * 0.25, 0.0, 0.7)))

	if show_velocity:
		for y in range(0, grid_height, 2):
			for x in range(0, grid_width, 2):
				var vi := field.idx(x, y)
				var vvx := field.out_vx[vi]
				var vvy := field.out_vy[vi]
				var mag := sqrt(vvx * vvx + vvy * vvy)
				if mag > 1.0:
					var origin := Vector2((float(x) + 0.5) * cs_val, (float(y) + 0.5) * cs_val)
					var tip := origin + Vector2(vvx, vvy) / mag * cs_val * 0.9
					draw_line(origin, tip, Color(0.3, 0.6, 1.0, 0.4), 1.0)

	if show_goal:
		var max_d := 1.0
		for i in range(grid_width * grid_height):
			if field.goal_dist[i] < 1e5:
				max_d = maxf(max_d, field.goal_dist[i])
		for y in range(grid_height):
			for x in range(grid_width):
				var gi := field.idx(x, y)
				if field.terrain[gi] > 0.5:
					continue
				var t := 1.0 - field.goal_dist[gi] / max_d
				draw_rect(Rect2(x * cs_val, y * cs_val, cs_val, cs_val),
						  Color(0.05, t * 0.5, t * 0.35, 0.35))

	if selected_agent >= 0 and selected_agent < display_pos.size():
		var sel_pos := display_pos[selected_agent]
		var sel_alive := cached_info.size() > selected_agent and (cached_info[selected_agent] & 1) != 0
		var sel_col := Color(1.0, 1.0, 0.2, 0.8) if sel_alive else Color(1.0, 0.3, 0.3, 0.6)

		var sel_info := cached_info[selected_agent] if selected_agent < cached_info.size() else 0
		var sel_fac := (sel_info >> 1) & 0x1F
		var sel_grp := faction_to_group[sel_fac] if sel_fac < faction_to_group.size() else 0
		var grp_bit := 1 << sel_grp

		if cached_cell_blocked.size() > 0:
			var total_cells := grid_width * grid_height
			for ci in range(mini(total_cells, cached_cell_blocked.size())):
				var cb := cached_cell_blocked[ci]
				if cb == -1:
					continue
				if (cb & grp_bit) == 0:
					continue
				var cx := ci % grid_width
				var cy := ci / grid_width
				draw_rect(Rect2(cx * cs_val, cy * cs_val, cs_val, cs_val),
						  Color(1.0, 0.2, 0.2, 0.35))

		draw_arc(sel_pos, agent_radius * 2.5, 0, TAU, 32, sel_col, 2.0)
		draw_arc(sel_pos, agent_radius * 3.5, 0, TAU, 32, Color(sel_col.r, sel_col.g, sel_col.b, 0.3), 1.0)
		if selected_agent < cpu_atk_range.size():
			var rng := cpu_atk_range[selected_agent]
			if rng > 0.0:
				draw_arc(sel_pos, rng, 0, TAU, 64, Color(1.0, 0.4, 0.2, 0.5), 1.5)
		if nearest_enemy >= 0 and nearest_enemy < display_pos.size():
			var ne_pos := display_pos[nearest_enemy]
			draw_line(sel_pos, ne_pos, Color(1.0, 0.2, 0.2, 0.7), 1.5)
			draw_arc(ne_pos, agent_radius * 2.0, 0, TAU, 24, Color(1.0, 0.2, 0.2, 0.8), 2.0)

		var sel_is_disp := (sel_info & (1 << 10)) != 0
		if sel_is_disp and cached_cell_atk.size() > 0:
			var sel_cx := clampi(int(agent_pos[selected_agent].x / cell_size), 0, grid_width - 1)
			var sel_cy := clampi(int(agent_pos[selected_agent].y / cell_size), 0, grid_height - 1)
			var ci := sel_cy * grid_width + sel_cx
			var blocker := cached_cell_atk[ci] if ci < cached_cell_atk.size() else -1
			draw_rect(Rect2(sel_cx * cs_val, sel_cy * cs_val, cs_val, cs_val),
					  Color(1.0, 0.5, 0.0, 0.5))
			if blocker >= 0 and blocker != selected_agent and blocker < display_pos.size():
				var bpos := display_pos[blocker]
				draw_line(sel_pos, bpos, Color(1.0, 0.6, 0.0, 0.8), 2.0)
				draw_arc(bpos, agent_radius * 2.0, 0, TAU, 24, Color(1.0, 0.6, 0.0, 0.9), 2.0)

	if brush_mode != BrushMode.NONE:
		var mpos := get_global_mouse_position()
		var col: Color
		match brush_mode:
			BrushMode.WALL:  col = Color(0.4, 0.8, 1.0, 0.35)
			BrushMode.GOAL:  col = Color(0.3, 1.0, 0.4, 0.35)
			BrushMode.ERASE: col = Color(1.0, 0.4, 0.4, 0.35)
			_:               col = Color(1, 1, 1, 0.2)
		draw_arc(mpos, brush_radius * cs_val, 0, TAU, 32, col, 1.5)
		var cx := int(mpos.x / cs_val)
		var cy := int(mpos.y / cs_val)
		var r := brush_radius
		for dy in range(-r, r + 1):
			for dx in range(-r, r + 1):
				if dx * dx + dy * dy > r * r:
					continue
				var gx := cx + dx
				var gy := cy + dy
				if gx >= 0 and gx < grid_width and gy >= 0 and gy < grid_height:
					draw_rect(Rect2(gx * cs_val, gy * cs_val, cs_val, cs_val),
							  Color(col.r, col.g, col.b, 0.15))


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_SPACE: _toggle_pause()
			KEY_R: _reset_sim()
			KEY_D:
				show_density = not show_density
				btn_density.button_pressed = show_density; queue_redraw()
			KEY_V:
				show_velocity = not show_velocity
				btn_velocity.button_pressed = show_velocity; queue_redraw()
			KEY_G:
				show_goal = not show_goal
				btn_goal.button_pressed = show_goal; queue_redraw()
			KEY_L:
				if DEBUG: _dump_jitter_log()
			KEY_1: _set_brush(BrushMode.NONE)
			KEY_2: _set_brush(BrushMode.WALL)
			KEY_3: _set_brush(BrushMode.GOAL)
			KEY_4: _set_brush(BrushMode.ERASE)

	if brush_mode == BrushMode.NONE:
		if event is InputEventMouseButton:
			if event.pressed:
				if event.button_index == MOUSE_BUTTON_LEFT:
					_pick_agent(get_global_mouse_position())
				elif event.button_index == MOUSE_BUTTON_RIGHT:
					selected_agent = -1
					nearest_enemy = -1
					_update_select_label()
					queue_redraw()
				elif event.button_index == MOUSE_BUTTON_WHEEL_UP:
					var mouse_world := get_global_mouse_position()
					cam_zoom = minf(cam_zoom * 1.15, 5.0)
					cam.zoom = Vector2(cam_zoom, cam_zoom)
					var new_mouse_world := get_global_mouse_position()
					cam.position += mouse_world - new_mouse_world
				elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
					var mouse_world := get_global_mouse_position()
					cam_zoom = maxf(cam_zoom / 1.15, 0.25)
					cam.zoom = Vector2(cam_zoom, cam_zoom)
					var new_mouse_world := get_global_mouse_position()
					cam.position += mouse_world - new_mouse_world
			if event.button_index == MOUSE_BUTTON_MIDDLE:
				cam_dragging = event.pressed
				if cam_dragging:
					cam_drag_origin = event.position
		if event is InputEventMouseMotion and cam_dragging:
			cam.position -= (event.position - cam_drag_origin) / cam_zoom
			cam_drag_origin = event.position
		return

	if event is InputEventMouseButton:
		if event.button_index == MOUSE_BUTTON_LEFT or event.button_index == MOUSE_BUTTON_RIGHT:
			is_painting = event.pressed
			paint_erase = (event.button_index == MOUSE_BUTTON_RIGHT)
			if is_painting:
				_apply_brush(get_global_mouse_position())
		elif event.pressed:
			if event.button_index == MOUSE_BUTTON_WHEEL_UP:
				brush_radius = mini(brush_radius + 1, 8); queue_redraw()
			elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
				brush_radius = maxi(brush_radius - 1, 1); queue_redraw()

	if event is InputEventMouseMotion and is_painting:
		_apply_brush(get_global_mouse_position())
		queue_redraw()


func _set_brush(mode: BrushMode) -> void:
	brush_mode = mode
	var lbl := get_node_or_null("CanvasLayer/PanelContainer/VBoxContainer/BrushLabel") as Label
	if lbl:
		lbl.text = "Mode: %s | Radius: %d" % [_brush_name(), brush_radius]
	queue_redraw()


#region debug — jitter detection / crossing detection / DISP persistence monitor

func _collect_all_frame(dt: float) -> Dictionary:
	var frame := {}
	for id in range(agent_count):
		if id >= cached_info.size():
			break
		var ainfo := cached_info[id]
		if (ainfo & 1) == 0:
			continue
		var pos := display_pos[id] if id < display_pos.size() else Vector2.ZERO
		var prev: Vector2 = _dbg_prev_pos.get(id, pos)
		var eff_vel := (pos - prev) / maxf(dt, 0.001)
		_dbg_prev_pos[id] = pos
		frame[id] = {
			"alive": true,
			"px": pos.x, "py": pos.y,
			"vx": eff_vel.x, "vy": eff_vel.y,
			"spd": eff_vel.length(),
			"atk": (ainfo & (1 << 9)) != 0,
			"disp": (ainfo & (1 << 10)) != 0,
			"fac": (ainfo >> 1) & 0x1F,
			"dmg": cached_damage[id] if id < cached_damage.size() else 0,
		}
	return frame


func _log_bow_frame(dt: float) -> void:
	if cached_info.is_empty():
		return

	var cur_frame := _collect_all_frame(dt)

	if not _dbg_recording:
		# maintain pre-buffer
		_dbg_pre_buffer.append(cur_frame)
		if _dbg_pre_buffer.size() > _DBG_PRE_FRAMES:
			_dbg_pre_buffer.pop_front()

		_check_crossing(cur_frame)

		# scan ATK agents for jitter + visual marking
		_dbg_jitter_marked.clear()
		for id in cur_frame:
			var d: Dictionary = cur_frame[id]
			if not d.atk:
				_dbg_jitter_frames.erase(id)
				continue
			if d.spd > _DBG_JITTER_SPD:
				_dbg_jitter_frames[id] = _dbg_jitter_frames.get(id, 0) + 1
			else:
				_dbg_jitter_frames[id] = 0
			if _dbg_jitter_frames[id] >= _DBG_JITTER_TRIGGER:
				_dbg_jitter_marked[id] = true
				if not _dbg_recording:
					_start_jitter_recording(id)

		# DISP persistence monitor
		if not _dbg_disp_recording:
			for id2 in cur_frame:
				var d2: Dictionary = cur_frame[id2]
				if d2.disp:
					_dbg_disp_frames[id2] = _dbg_disp_frames.get(id2, 0) + 1
				else:
					_dbg_disp_frames.erase(id2)
				if _dbg_disp_frames.get(id2, 0) >= _DBG_DISP_TRIGGER:
					_dbg_disp_recording = true
					_dbg_disp_agent = id2
					_dbg_disp_log.clear()
					print("=== LONG DISP: Agent #%d stuck for %d+ frames ===" % [id2, _DBG_DISP_TRIGGER])
					break
		else:
			if cur_frame.has(_dbg_disp_agent):
				var dd: Dictionary = cur_frame[_dbg_disp_agent]
				var cx := clampi(int(dd.px / cell_size), 0, grid_width - 1)
				var cy := clampi(int(dd.py / cell_size), 0, grid_height - 1)
				var ci := cy * grid_width + cx
				var blocker := cached_cell_atk[ci] if ci < cached_cell_atk.size() else -1
				_dbg_disp_log.append({
					"px": dd.px, "py": dd.py, "vx": dd.vx, "vy": dd.vy,
					"spd": dd.spd, "cell": Vector2i(cx, cy), "blocker": blocker,
					"disp": dd.disp, "atk": dd.atk
				})
			else:
				print("=== DISP LOG: Agent #%d lost (died/despawned) during recording ===" % _dbg_disp_agent)
				_dump_disp_log()
				_dbg_disp_recording = false
				_dbg_disp_frames.erase(_dbg_disp_agent)
				_dbg_disp_agent = -1
			if _dbg_disp_log.size() >= _DBG_DISP_LOG_FRAMES:
				_dump_disp_log()
				_dbg_disp_recording = false
				_dbg_disp_frames.erase(_dbg_disp_agent)
				_dbg_disp_agent = -1
	else:
		# recording phase: filter to tracked agents
		var filtered := {}
		for id in _dbg_jitter_agents:
			if cur_frame.has(id):
				filtered[id] = cur_frame[id]
		_dbg_jitter_log.append(filtered)
		if _dbg_jitter_log.size() >= _DBG_MAX_FRAMES:
			_dump_jitter_log()
			_dbg_recording = false
			_dbg_jitter_log.clear()
			_dbg_jitter_agents = PackedInt32Array()
			_dbg_jitter_frames.clear()


func _dump_disp_log() -> void:
	if _dbg_disp_log.is_empty():
		return
	print("=== DISP LOG (Agent #%d, %d frames) ===" % [_dbg_disp_agent, _dbg_disp_log.size()])
	for f in range(_dbg_disp_log.size()):
		var d: Dictionary = _dbg_disp_log[f]
		var flags := ""
		if d.atk: flags += "ATK "
		if d.disp: flags += "DISP "
		print("F%02d: p=(%.1f,%.1f) v=(%.1f,%.1f) spd=%.1f cell=(%d,%d) blocker=#%d %s" % [
			f, d.px, d.py, d.vx, d.vy, d.spd,
			d.cell.x, d.cell.y, d.blocker, flags])
	print("=== END DISP LOG ===")
	_dbg_disp_log.clear()


func _check_crossing(cur_frame: Dictionary) -> void:
	_dbg_cross_timer += 1
	if _dbg_cross_timer < _DBG_CROSS_INTERVAL:
		return
	_dbg_cross_timer = 0
	var half := NUM_FACTIONS / 2
	var atk_list: Array[int] = []
	var non_atk_list: Array[int] = []
	for id in cur_frame:
		var d: Dictionary = cur_frame[id]
		if d.atk:
			atk_list.append(id)
		else:
			non_atk_list.append(id)
	var cross_count := 0
	var examples: Array[String] = []
	var dsq_thresh := _DBG_CROSS_DIST * _DBG_CROSS_DIST
	for na in non_atk_list:
		var nd: Dictionary = cur_frame[na]
		var na_side := 0 if nd.fac < half else 1
		var na_pos := Vector2(nd.px, nd.py)
		for atk in atk_list:
			var ad: Dictionary = cur_frame[atk]
			var atk_side := 0 if ad.fac < half else 1
			if na_side == atk_side:
				continue
			var dsq := na_pos.distance_squared_to(Vector2(ad.px, ad.py))
			if dsq < dsq_thresh:
				cross_count += 1
				if examples.size() < 3:
					examples.append("#%d(f%d) near enemy ATK #%d(f%d) d=%.1f" % [na, nd.fac, atk, ad.fac, sqrt(dsq)])
				break
	if cross_count > 0:
		print("=== CROSS DETECTED: %d non-ATK within %.0fpx of enemy ATK ===" % [cross_count, _DBG_CROSS_DIST])
		for e in examples:
			print("  ", e)


func _start_jitter_recording(trigger_id: int) -> void:
	_dbg_recording = true

	var ids: Array[int] = [trigger_id]
	var trigger_pos := agent_pos[trigger_id] if trigger_id < agent_pos.size() else Vector2.ZERO

	var neighbors: Array = []
	for j in range(agent_count):
		if j == trigger_id or j >= cached_info.size():
			continue
		if (cached_info[j] & 1) == 0:
			continue
		var d := agent_pos[j].distance_squared_to(trigger_pos)
		neighbors.append({"id": j, "dsq": d})
	neighbors.sort_custom(func(a, b): return a.dsq < b.dsq)
	for k in range(mini(neighbors.size(), 14)):
		ids.append(neighbors[k].id)

	_dbg_jitter_agents = PackedInt32Array(ids)

	# seed log with pre-buffer frames (filtered to tracked agents)
	_dbg_jitter_log.clear()
	for pf: Dictionary in _dbg_pre_buffer:
		var filtered := {}
		for id in _dbg_jitter_agents:
			if pf.has(id):
				filtered[id] = pf[id]
		_dbg_jitter_log.append(filtered)
	_dbg_pre_buffer.clear()

	print("=== JITTER DETECTED: Agent #%d (spd>%.1f for %d+ frames) ===" % [
		trigger_id, _DBG_JITTER_SPD, _DBG_JITTER_TRIGGER])
	print("Recording %d frames (%d pre-buffered) for agents: %s" % [
		_DBG_MAX_FRAMES, _dbg_jitter_log.size(), str(_dbg_jitter_agents)])


func _dump_jitter_log() -> void:
	if _dbg_jitter_log.is_empty():
		print("No jitter log data.")
		return
	print("=== JITTER LOG (%d frames, tracking %s) ===" % [_dbg_jitter_log.size(), str(_dbg_jitter_agents)])
	for f in range(_dbg_jitter_log.size()):
		var fd: Dictionary = _dbg_jitter_log[f]
		var parts := PackedStringArray()
		for id in _dbg_jitter_agents:
			if not fd.has(id):
				continue
			var d: Dictionary = fd[id]
			if not d.alive:
				parts.append("#%d DEAD" % id)
				continue
			var flags := ""
			if d.atk: flags += "ATK "
			if d.disp: flags += "DISP "
			parts.append("#%d p=(%.1f,%.1f) v=(%.1f,%.1f) spd=%.1f %sdmg=%d" % [
				id, d.px, d.py, d.vx, d.vy, d.spd, flags, d.dmg
			])
		print("F%02d: %s" % [f, " | ".join(parts)])
	print("=== END JITTER LOG ===")

#endregion
