extends Node2D

const MAX_AGENTS := 100000
const NUM_FACTIONS := 32
const NUM_GOAL_GROUPS := 32

@export var grid_width := 800
@export var grid_height := 480
@export var cell_size := 8.0
@export var agent_count := 20000
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
const DISPLAY_LERP := 0.4

var world_size: Vector2
var goal_cells: PackedVector2Array
var _wall_sprite: Sprite2D
var _wall_tex_dirty := true

# GPU overlay system
var _ov_shd: RID
var _ov_pip: RID
var _ov_uset: RID
var _ov_buf_rgba: RID
var _ov_sprite: Sprite2D
var _ov_active_mode := -1

var swe_accum := 0.0
var swe_interval := 0.033

var show_density := false
var show_velocity := false
var paused := false

# Brush painting
enum BrushMode { NONE, WALL, ERASE }
var brush_mode: BrushMode = BrushMode.NONE
var brush_radius := 2
var is_painting := false
var paint_erase := false
var terrain_dirty := false

var perf_gpu := 0.0
var perf_read := 0.0
var perf_disppos := 0.0
var perf_mesh := 0.0
var perf_mm_readback_info := 0.0
var perf_mm_readback_dmg := 0.0
var perf_mm_loop := 0.0
var perf_mm_set_buffer := 0.0
var perf_goal := 0.0
var perf_alpha := 0.15
var perf_peak_total := 0.0
var perf_peak_detail := ""
var perf_render := 0.0
var _last_frame_usec := 0
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
var _goal_rr_idx := 0
var cached_info := PackedInt32Array()
var cached_damage := PackedInt32Array()
var cached_cell_atk := PackedInt32Array()
var cached_cell_blocked := PackedInt32Array()

var _agent_debounce_timer: Timer
var _agent_pending_count := -1

# HUD
var fps_label: Label
var hud_label: Label
var slider_agents: HSlider
var label_agents: Label
var btn_density: CheckButton
var btn_velocity: CheckButton
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
	agents.upload_faction_colors(faction_colors)
	agents.upload_display_pos(agent_pos, agent_count)

	_setup_multimesh()
	_setup_overlay()
	_setup_hud()
	_play_bgm()


func _exit_tree() -> void:
	_cleanup_overlay()


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

var castle_image_path := "res://castle_bw.png"
var castle_size := 50
var castle_centers: Array[Vector2i] = []
var castle_exits: Array[Vector2i] = []

func _build_environment() -> void:
	field.add_wall(0, 0, grid_width, 1)
	field.add_wall(0, grid_height - 1, grid_width, grid_height)
	field.add_wall(0, 0, 1, grid_height)
	field.add_wall(grid_width - 1, 0, grid_width, grid_height)

	_gen_castles()

	_wall_sprite = Sprite2D.new()
	_wall_sprite.centered = false
	_wall_sprite.z_index = 0
	_wall_sprite.scale = Vector2(cell_size, cell_size)
	_wall_sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	add_child(_wall_sprite)
	_wall_tex_dirty = true
	_setup_goals()


func _gen_castles() -> void:
	var FOOT := castle_size

	castle_centers.clear()
	castle_exits.clear()

	var castle_base: Image
	if castle_image_path.begins_with("res://"):
		var tex := load(castle_image_path) as Texture2D
		castle_base = tex.get_image() if tex else Image.new()
	else:
		castle_base = Image.new()
		castle_base.load(castle_image_path)
	castle_base.convert(Image.FORMAT_L8)
	castle_base.resize(FOOT, FOOT, Image.INTERPOLATE_NEAREST)

	var mcx := grid_width / 2
	var mcy := grid_height / 2
	var occupied: Array[Rect2i] = []

	for _attempt in range(5000):
		if castle_centers.size() >= NUM_FACTIONS:
			break
		var rx := randi_range(3, grid_width - FOOT - 3)
		var ry := randi_range(3, grid_height - FOOT - 3)
		var guard := Rect2i(rx - 4, ry - 4, FOOT + 8, FOOT + 8)
		var overlap := false
		for o in occupied:
			if guard.intersects(o):
				overlap = true
				break
		if overlap:
			continue
		occupied.append(Rect2i(rx, ry, FOOT, FOOT))

		var castle_img := castle_base.duplicate()
		var rot := randi_range(0, 3)
		match rot:
			1: castle_img.rotate_90(CLOCKWISE)
			2: castle_img.rotate_180()
			3: castle_img.rotate_90(COUNTERCLOCKWISE)
		if randf() < 0.5:
			castle_img.flip_x()

		for ly in range(FOOT):
			for lx in range(FOOT):
				if castle_img.get_pixel(lx, ly).get_luminance() > 0.5:
					var gx := rx + lx
					var gy := ry + ly
					if gx >= 0 and gx < grid_width and gy >= 0 and gy < grid_height:
						field.terrain[gy * grid_width + gx] = 1.0

		castle_centers.append(Vector2i(rx + FOOT / 2, ry + FOOT / 2))

		var best_exit := Vector2i(rx + FOOT / 2, ry)
		var best_dist := INF
		var center_to_map := Vector2(float(mcx - (rx + FOOT / 2)), float(mcy - (ry + FOOT / 2)))
		for lx in range(FOOT):
			for ly in [0, FOOT - 1]:
				if castle_img.get_pixel(lx, ly).get_luminance() < 0.5:
					var d := center_to_map.dot(Vector2(float(lx - FOOT / 2), float(ly - FOOT / 2)))
					if d > best_dist:
						continue
					best_dist = d
					best_exit = Vector2i(rx + lx, ry + ly)
		for ly in range(FOOT):
			for lx in [0, FOOT - 1]:
				if castle_img.get_pixel(lx, ly).get_luminance() < 0.5:
					var d := center_to_map.dot(Vector2(float(lx - FOOT / 2), float(ly - FOOT / 2)))
					if d > best_dist:
						continue
					best_dist = d
					best_exit = Vector2i(rx + lx, ry + ly)
		castle_exits.append(best_exit)


func _setup_goals() -> void:
	goal_cells = PackedVector2Array()

	var center_goal: Array[Vector2i] = [Vector2i(grid_width / 2, grid_height / 2)]
	group_goals = []
	for _g in range(NUM_GOAL_GROUPS):
		group_goals.append(center_goal)
	field.build_all_goal_fields(group_goals)



# ── Agents ───────────────────────────────────────────────────────────────

func _spawn_agents() -> void:
	agent_pos = PackedVector2Array(); agent_pos.resize(agent_count)
	agent_vel = PackedVector2Array(); agent_vel.resize(agent_count)
	display_pos = PackedVector2Array(); display_pos.resize(agent_count)
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
				(float(cc.x) + randf_range(-20.0, 20.0)) * cell_size,
				(float(cc.y) + randf_range(-20.0, 20.0)) * cell_size)
		else:
			agent_pos[i] = Vector2(
				randf_range(cell_size * 3.0, world_size.x - cell_size * 3.0),
				randf_range(cell_size * 3.0, world_size.y - cell_size * 3.0))
		agent_vel[i] = Vector2.ZERO
		display_pos[i] = agent_pos[i]

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

func _rebuild_wall_texture() -> void:
	var img := Image.create(grid_width, grid_height, false, Image.FORMAT_RGBA8)
	img.fill(Color(0, 0, 0, 0))
	var t := field.terrain
	var w := grid_width
	for y in range(grid_height):
		var row := y * w
		for x in range(w):
			if t[row + x] > 0.5:
				img.set_pixel(x, y, Color(0.35, 0.35, 0.42, 1.0))
	_wall_sprite.texture = ImageTexture.create_from_image(img)
	_wall_tex_dirty = false



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
					changed = true
					terrain_dirty = true

			elif brush_mode == BrushMode.ERASE:
				if field.terrain[gi] > 0.5:
					field.terrain[gi] = 0.0
					changed = true
					terrain_dirty = true

	if changed:
		queue_redraw()


func _flush_paint() -> void:
	if terrain_dirty:
		field._upload(field.buf_terrain, field.terrain)
		_wall_tex_dirty = true
		terrain_dirty = false
	if _wall_tex_dirty:
		_rebuild_wall_texture()


# ── MultiMesh ────────────────────────────────────────────────────────────

func _setup_multimesh() -> void:
	var quad := QuadMesh.new()
	quad.size = Vector2(agent_radius * 2.0, agent_radius * 2.0)
	multi_mesh = MultiMesh.new()
	multi_mesh.transform_format = MultiMesh.TRANSFORM_2D
	multi_mesh.use_colors = true
	multi_mesh.instance_count = agent_count
	multi_mesh.visible_instance_count = agent_count
	multi_mesh.mesh = quad
	mm_instance = MultiMeshInstance2D.new()
	mm_instance.multimesh = multi_mesh
	mm_instance.texture = _make_circle_texture(12)
	add_child(mm_instance)


# ── GPU Overlay ──────────────────────────────────────────────────────────

func _setup_overlay() -> void:
	var spirv := (load("res://shaders/build_overlay.glsl") as RDShaderFile).get_spirv()
	_ov_shd = rd.shader_create_from_spirv(spirv)
	_ov_pip = rd.compute_pipeline_create(_ov_shd)

	var n_bytes := grid_width * grid_height * 4
	var z := PackedByteArray(); z.resize(n_bytes); z.fill(0)
	_ov_buf_rgba = rd.storage_buffer_create(n_bytes, z)

	var _u := func(binding: int, buf: RID) -> RDUniform:
		var u := RDUniform.new()
		u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		u.binding = binding
		u.add_id(buf)
		return u

	_ov_uset = rd.uniform_set_create([
		_u.call(0, field.buf_density),
		_u.call(1, field.buf_terrain),
		_u.call(2, field.buf_goal_dist),
		_u.call(3, agents.buf_cell_blocked),
		_u.call(4, _ov_buf_rgba),
		_u.call(5, field.buf_out_vx),
		_u.call(6, field.buf_out_vy),
		_u.call(7, agents.buf_faction_presence),
	], _ov_shd, 0)

	_ov_sprite = Sprite2D.new()
	_ov_sprite.centered = false
	_ov_sprite.z_index = 1
	_ov_sprite.scale = Vector2(cell_size, cell_size)
	_ov_sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	_ov_sprite.visible = false
	add_child(_ov_sprite)


func _dispatch_overlay(mode: int, param_i: int = 0, param_f: float = 0.0, group_off: int = 0, fac_mask: int = 0) -> void:
	var p := PackedByteArray(); p.resize(32)
	p.encode_s32(0, grid_width)
	p.encode_s32(4, grid_height)
	p.encode_s32(8, mode)
	p.encode_s32(12, param_i)
	p.encode_float(16, param_f)
	p.encode_s32(20, group_off)
	p.encode_s32(24, fac_mask)

	var cl := rd.compute_list_begin()
	rd.compute_list_bind_compute_pipeline(cl, _ov_pip)
	rd.compute_list_bind_uniform_set(cl, _ov_uset, 0)
	rd.compute_list_set_push_constant(cl, p, 32)
	rd.compute_list_dispatch(cl, ceili(float(grid_width) / 8.0), ceili(float(grid_height) / 8.0), 1)
	rd.compute_list_end()
	rd.submit()
	rd.sync()

	var raw := rd.buffer_get_data(_ov_buf_rgba, 0, grid_width * grid_height * 4)
	var img := Image.create_from_data(grid_width, grid_height, false, Image.FORMAT_RGBA8, raw)
	_ov_sprite.texture = ImageTexture.create_from_image(img)
	_ov_sprite.visible = true
	_ov_active_mode = mode


func _hide_overlay() -> void:
	if _ov_sprite:
		_ov_sprite.visible = false
	_ov_active_mode = -1


func _cleanup_overlay() -> void:
	if rd == null:
		return
	for rid in [_ov_uset, _ov_pip, _ov_shd, _ov_buf_rgba]:
		if rid.is_valid():
			rd.free_rid(rid)


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


func _sync_multimesh_gpu() -> void:
	var ta := Time.get_ticks_usec()
	mm_buf = agents.readback_mm_buf(agent_count)
	var tb := Time.get_ticks_usec()
	alive_count = agents.readback_alive_count()
	if selected_agent >= 0:
		cached_info = agents.readback_agent_info()
		cached_damage = agents.readback_damage_acc()
		cached_cell_atk = agents.readback_cell_attacker()
		cached_cell_blocked = agents.readback_cell_blocked()
	multi_mesh.set_buffer(mm_buf)
	var tc := Time.get_ticks_usec()
	var alpha := perf_alpha
	perf_mm_readback_info = lerpf(perf_mm_readback_info, float(tb - ta) / 1000.0, alpha)
	perf_mm_readback_dmg  = 0.0
	perf_mm_loop          = 0.0
	perf_mm_set_buffer    = lerpf(perf_mm_set_buffer, float(tc - tb) / 1000.0, alpha)


func _sync_multimesh_cpu() -> void:
	var ta := Time.get_ticks_usec()
	cached_info = agents.readback_agent_info()
	var tb := Time.get_ticks_usec()
	cached_damage = agents.readback_damage_acc()
	var tc := Time.get_ticks_usec()
	if selected_agent >= 0:
		cached_cell_atk = agents.readback_cell_attacker()
		cached_cell_blocked = agents.readback_cell_blocked()
	alive_count = 0
	var stride := 12
	var buf_size := agent_count * stride
	if mm_buf.size() != buf_size:
		mm_buf.resize(buf_size)
	var info := cached_info
	var dmg := cached_damage
	var hp := cpu_max_hp
	var dpos := display_pos
	var colors := faction_colors
	var cnt := agent_count
	var alive := 0
	for i in range(cnt):
		var off := i * stride
		var ainfo := info[i]
		if (ainfo & 1) != 0:
			var fac := (ainfo >> 1) & 0x1F
			var inv_hp := 1.0 / maxf(float(hp[i]), 1.0)
			var brightness := lerpf(0.2, 1.0, 1.0 - clampf(float(dmg[i]) * inv_hp, 0.0, 1.0))
			var c: Color = colors[fac]
			var p: Vector2 = dpos[i]
			mm_buf[off]     = 1.0;  mm_buf[off + 1] = 0.0
			mm_buf[off + 2] = 0.0;  mm_buf[off + 3] = p.x
			mm_buf[off + 4] = 0.0;  mm_buf[off + 5] = 1.0
			mm_buf[off + 6] = 0.0;  mm_buf[off + 7] = p.y
			mm_buf[off + 8]  = c.r * brightness
			mm_buf[off + 9]  = c.g * brightness
			mm_buf[off + 10] = c.b * brightness
			mm_buf[off + 11] = 1.0
			alive += 1
		else:
			mm_buf[off]     = 1.0;  mm_buf[off + 1] = 0.0
			mm_buf[off + 2] = 0.0;  mm_buf[off + 3] = -10000.0
			mm_buf[off + 4] = 0.0;  mm_buf[off + 5] = 1.0
			mm_buf[off + 6] = 0.0;  mm_buf[off + 7] = -10000.0
			mm_buf[off + 8]  = 0.0;  mm_buf[off + 9]  = 0.0
			mm_buf[off + 10] = 0.0;  mm_buf[off + 11] = 0.0
	var td := Time.get_ticks_usec()
	alive_count = alive
	multi_mesh.set_buffer(mm_buf)
	var te := Time.get_ticks_usec()
	var a := perf_alpha
	perf_mm_readback_info = lerpf(perf_mm_readback_info, float(tb - ta) / 1000.0, a)
	perf_mm_readback_dmg  = lerpf(perf_mm_readback_dmg,  float(tc - tb) / 1000.0, a)
	perf_mm_loop          = lerpf(perf_mm_loop,          float(td - tc) / 1000.0, a)
	perf_mm_set_buffer    = lerpf(perf_mm_set_buffer,    float(te - td) / 1000.0, a)


func _pick_agent(world_pos: Vector2) -> void:
	agent_pos = agents.readback_current_positions()
	cached_info = agents.readback_agent_info()
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

func _process(dt: float) -> void:
	var now_usec := Time.get_ticks_usec()
	if _last_frame_usec > 0:
		perf_render = lerpf(perf_render, float(now_usec - _last_frame_usec) / 1000.0, perf_alpha)
	_last_frame_usec = now_usec

	_flush_paint()
	if paused:
		return

	var t0 := Time.get_ticks_usec()
	agents.reset_alive_counter()

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

	# 5 — Velocity field (multi-group)
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
	rd.compute_list_add_barrier(cl)

	# 7 — Build MultiMesh buffer on GPU
	agents.dispatch_build_mm(cl, DISPLAY_LERP)

	rd.compute_list_end()
	rd.submit()
	rd.sync()
	var t1 := Time.get_ticks_usec()

	# 8 — Swap double-buffer; readback positions only when needed
	agents.swap_buffers()
	var t1b := Time.get_ticks_usec()
	if selected_agent >= 0:
		agent_pos = agents.readback_current_positions()
	var t1c := Time.get_ticks_usec()
	_sync_multimesh_gpu()
	var t2 := Time.get_ticks_usec()
	_update_select_label()

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

	# 10 — GPU overlay (density / velocity), filtered by selected agent's faction
	var ov_group_off := 0
	var ov_fac_mask := 0
	if selected_agent >= 0 and selected_agent < cached_info.size():
		var sel_info := cached_info[selected_agent]
		var sel_fac := int((sel_info >> 1) & 0x1F)
		var sel_grp := faction_to_group[sel_fac] if sel_fac < faction_to_group.size() else 0
		ov_group_off = sel_grp * grid_width * grid_height
		ov_fac_mask = 1 << sel_fac
	if show_density:
		_dispatch_overlay(0, 0, 0.0, ov_group_off, ov_fac_mask)
	elif show_velocity:
		_dispatch_overlay(3, 0, 0.0, ov_group_off, ov_fac_mask)
	elif _ov_active_mode >= 0:
		_hide_overlay()

	var a := perf_alpha
	var gpu_ms := float(t1 - t0) / 1000.0
	var rd_ms  := float(t1c - t1b) / 1000.0
	var mm_ms  := float(t2 - t1c) / 1000.0
	perf_gpu     = lerpf(perf_gpu,     gpu_ms, a)
	perf_read    = lerpf(perf_read,    rd_ms, a)
	perf_disppos = 0.0
	perf_mesh    = lerpf(perf_mesh,    mm_ms, a)
	var total := perf_gpu + perf_read + perf_mesh
	var frame_total := float(t3 - t0) / 1000.0
	if frame_total > perf_peak_total:
		perf_peak_total = frame_total
		perf_peak_detail = "GPU=%.1f Rd=%.1f MM=%.1f G=%.1f" % [
			gpu_ms, rd_ms, mm_ms, goal_ms]
	if frame_total > 50.0:
		print("[SPIKE] %.1f ms | GPU=%.1f Rd=%.1f MM=%.1f Goal=%.1f" % [
			frame_total, gpu_ms, rd_ms, mm_ms, goal_ms])
	fps_label.text = "FPS: %d" % Engine.get_frames_per_second()
	hud_label.text = (
		"GPU %.1f  Rd %.1f  MM %.1f  Goal %.1f | Alive: %d / %d\n" % [perf_gpu, perf_read, perf_mesh, perf_goal, alive_count, agent_count]
		+ "PEAK: %.1f ms  %s" % [perf_peak_total, perf_peak_detail]
	)

	if brush_mode != BrushMode.NONE or selected_agent >= 0:
		queue_redraw()


# ── BGM ──────────────────────────────────────────────────────────────────

func _play_bgm() -> void:
	var stream := load("res://audio/soviet_march.mp3") as AudioStream
	if stream == null:
		return
	var player := AudioStreamPlayer.new()
	player.stream = stream
	player.volume_db = -6.0
	player.bus = "Master"
	add_child(player)
	player.play()
	player.finished.connect(player.play)


# ── HUD ──────────────────────────────────────────────────────────────────

func _setup_hud() -> void:
	_agent_debounce_timer = Timer.new()
	_agent_debounce_timer.one_shot = true
	_agent_debounce_timer.wait_time = 1.0
	_agent_debounce_timer.timeout.connect(_apply_agent_count)
	add_child(_agent_debounce_timer)

	var canvas := CanvasLayer.new()
	canvas.layer = 100
	add_child(canvas)

	fps_label = Label.new()
	fps_label.position = Vector2(10, 8)
	fps_label.add_theme_color_override("font_color", Color(1.0, 1.0, 0.2))
	fps_label.add_theme_font_size_override("font_size", 24)
	canvas.add_child(fps_label)

	hud_label = Label.new()
	hud_label.position = Vector2(10, 38)
	hud_label.add_theme_color_override("font_color", Color(0.75, 0.75, 0.75))
	hud_label.add_theme_font_size_override("font_size", 11)
	canvas.add_child(hud_label)

	select_label = Label.new()
	select_label.position = Vector2(10, 74)
	select_label.add_theme_color_override("font_color", Color(1.0, 0.95, 0.6))
	select_label.add_theme_font_size_override("font_size", 11)
	canvas.add_child(select_label)

	var panel := PanelContainer.new()
	var vp_w := get_viewport().get_visible_rect().size.x
	panel.position = Vector2(vp_w - 270, 40)
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
	vbox.add_child(_lbl("Castle Template"))
	var castle_hbox := HBoxContainer.new()
	castle_hbox.add_theme_constant_override("separation", 4)
	var castle_name_label := Label.new()
	castle_name_label.name = "CastleNameLabel"
	castle_name_label.add_theme_font_size_override("font_size", 11)
	castle_name_label.add_theme_color_override("font_color", Color(0.8, 0.85, 1.0))
	castle_name_label.text = castle_image_path.get_file()
	castle_name_label.size_flags_horizontal = Control.SIZE_EXPAND_FILL
	castle_hbox.add_child(castle_name_label)
	var btn_browse := Button.new()
	btn_browse.text = "Browse"
	btn_browse.add_theme_font_size_override("font_size", 11)
	btn_browse.pressed.connect(func():
		var fd := FileDialog.new()
		fd.file_mode = FileDialog.FILE_MODE_OPEN_FILE
		fd.access = FileDialog.ACCESS_FILESYSTEM
		fd.filters = PackedStringArray(["*.png ; PNG Images"])
		fd.size = Vector2i(600, 400)
		fd.file_selected.connect(func(path: String):
			castle_image_path = path
			castle_name_label.text = path.get_file()
			fd.queue_free()
		)
		fd.canceled.connect(func(): fd.queue_free())
		get_tree().root.add_child(fd)
		fd.popup_centered()
	)
	castle_hbox.add_child(btn_browse)
	vbox.add_child(castle_hbox)
	var castle_size_lbl := _lbl("Size: %d" % castle_size)
	vbox.add_child(castle_size_lbl)
	var slider_castle := _slider(20, 120, castle_size, 5)
	slider_castle.value_changed.connect(func(v: float):
		castle_size = int(v)
		castle_size_lbl.text = "Size: %d" % castle_size
	)
	vbox.add_child(slider_castle)

	vbox.add_child(_sep())

	btn_density = _toggle("Density heatmap", show_density)
	btn_density.toggled.connect(func(on: bool):
		show_density = on
		if not on and not show_velocity: _hide_overlay()
	)
	vbox.add_child(btn_density)
	btn_velocity = _toggle("Velocity field", show_velocity)
	btn_velocity.toggled.connect(func(on: bool):
		show_velocity = on
		if not on and not show_density: _hide_overlay()
	)
	vbox.add_child(btn_velocity)

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
	vbox.add_child(_lbl("Brush  [1]None [2]Wall [3]Erase"))
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
		BrushMode.ERASE: return "Erase"
		_: return "None"

func _update_agent_label() -> void:
	label_agents.text = "Agents: %d" % agent_count
	label_agents.add_theme_color_override("font_color", Color(0.9, 0.75, 0.4))

func _on_agents_changed(val: float) -> void:
	_agent_pending_count = int(val)
	label_agents.text = "Agents: %d (pending...)" % _agent_pending_count
	_agent_debounce_timer.start()


func _apply_agent_count() -> void:
	if _agent_pending_count < 0:
		return
	agent_count = _agent_pending_count
	_agent_pending_count = -1
	selected_agent = -1
	nearest_enemy = -1
	_update_agent_label()
	multi_mesh.instance_count = agent_count
	multi_mesh.visible_instance_count = agent_count
	mm_buf.resize(agent_count * 12)
	_spawn_agents()
	agents.upload_agents(agent_pos, agent_vel, agent_count)
	_upload_combat()
	agents.upload_display_pos(agent_pos, agent_count)
	agents.clear_corpse_map()
	_sync_multimesh_cpu()

func _toggle_pause() -> void:
	paused = not paused
	btn_pause.text = "  Resume  " if paused else "  Pause  "
	queue_redraw()

func _reset_sim() -> void:
	selected_agent = -1
	nearest_enemy = -1
	field.reset_flux()
	field.terrain.fill(0.0)
	field.add_wall(0, 0, grid_width, 1)
	field.add_wall(0, grid_height - 1, grid_width, grid_height)
	field.add_wall(0, 0, 1, grid_height)
	field.add_wall(grid_width - 1, 0, grid_width, grid_height)
	_gen_castles()
	_wall_tex_dirty = true
	_spawn_agents()
	agents.upload_agents(agent_pos, agent_vel, agent_count)
	_upload_combat()
	agents.upload_display_pos(agent_pos, agent_count)
	agents.clear_corpse_map()
	_setup_goals()
	field.upload_static_data()
	_sync_multimesh_cpu()


# ── Drawing ──────────────────────────────────────────────────────────────

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, world_size), Color(0.11, 0.11, 0.14))
	var cs_val := cell_size

	for gc in goal_cells:
		draw_rect(Rect2(gc.x * cs_val, gc.y * cs_val, cs_val, cs_val), Color(0.2, 0.75, 0.35, 0.18))

	if selected_agent >= 0 and selected_agent < agent_pos.size():
		var sel_pos := agent_pos[selected_agent]
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
		if nearest_enemy >= 0 and nearest_enemy < agent_pos.size():
			var ne_pos := agent_pos[nearest_enemy]
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
			if blocker >= 0 and blocker != selected_agent and blocker < agent_pos.size():
				var bpos := agent_pos[blocker]
				draw_line(sel_pos, bpos, Color(1.0, 0.6, 0.0, 0.8), 2.0)
				draw_arc(bpos, agent_radius * 2.0, 0, TAU, 24, Color(1.0, 0.6, 0.0, 0.9), 2.0)

	if brush_mode != BrushMode.NONE:
		var mpos := get_global_mouse_position()
		var col: Color
		match brush_mode:
			BrushMode.WALL:  col = Color(0.4, 0.8, 1.0, 0.35)
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
				btn_density.button_pressed = show_density
				if not show_density and not show_velocity: _hide_overlay()
			KEY_V:
				show_velocity = not show_velocity
				btn_velocity.button_pressed = show_velocity
				if not show_velocity and not show_density: _hide_overlay()
			KEY_1: _set_brush(BrushMode.NONE)
			KEY_2: _set_brush(BrushMode.WALL)
			KEY_3: _set_brush(BrushMode.ERASE)

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
