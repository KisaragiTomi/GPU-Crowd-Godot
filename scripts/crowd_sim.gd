extends Node2D

class MinimapOverlay:
	extends Control

	var sim

	func _draw() -> void:
		if sim == null or sim.cam == null:
			return
		var map_rect := Rect2(Vector2.ZERO, size)
		draw_rect(map_rect, Color(0.02, 0.025, 0.03, 0.2), true)
		draw_rect(map_rect, Color(0.85, 0.9, 1.0, 0.9), false, 2.0)

		if sim.world_size.x <= 0.0 or sim.world_size.y <= 0.0:
			return
		var vp_size: Vector2 = sim.get_viewport().get_visible_rect().size
		var zoom: Vector2 = sim.cam.zoom
		var view_size := Vector2(vp_size.x / zoom.x, vp_size.y / zoom.y)
		var view_pos: Vector2 = sim.cam.global_position - view_size * 0.5
		var map_scale: Vector2 = size / sim.world_size
		var view_rect := Rect2(
			Vector2(view_pos.x * map_scale.x, view_pos.y * map_scale.y),
			Vector2(view_size.x * map_scale.x, view_size.y * map_scale.y)
		)
		view_rect = view_rect.intersection(map_rect.grow(-2.0))
		if view_rect.size.x <= 0.0 or view_rect.size.y <= 0.0:
			return
		draw_rect(view_rect, Color(1.0, 0.92, 0.25, 0.16), true)
		draw_rect(view_rect, Color(1.0, 0.92, 0.25, 0.95), false, 1.5)


class LaserOverlay:
	extends Node2D

	var sim
	var width := 1.0

	# 顶点色 alpha = fade，与 self_modulate.a 相乘得到本层的 alpha
	func _draw() -> void:
		if sim != null and not sim._laser_points.is_empty():
			draw_multiline_colors(sim._laser_points, sim._laser_colors, width, true)


const MAX_AGENTS := 100000
const NUM_FACTIONS := 32
const NUM_GOAL_GROUPS := 32
const CAM_PAN_SPEED := 800.0
const MINIMAP_SCREEN_FRACTION := 0.2
const MINIMAP_MARGIN := 12.0
const GI_REGION_SCALE := 1.2
const GI_PRECISION_MIN := 0.50
const GI_PRECISION_MAX := 1.00
const GI_PRECISION_STEP := 0.05
const GI_AGENT_EMIT_STRENGTH := 1.15
const GI_AGENT_SPLAT_RADIUS_SCALE := 1.0
const GI_DISPLAY_ALPHA := 0.60
const HIT_GI_FLASH_DECAY_PER_SEC := 12.0
const HIT_GI_FLASH_VALUE := 0.75
const RANGED_LASER_LIFE := 0.12
const RANGED_LASER_GI_WIDTH_WORLD := 0.45
const RANGED_LASER_GI_STRENGTH := 3.6
const RANGED_LASER_VIS_WIDTH := 0.28
const RANGED_LASER_VIS_GLOW_WIDTH := 0.78
const RANGED_LASER_VIS_SOFT_WIDTH := 1.55
const RENDER_LAYER_WORLD := 1 << 0
const RENDER_LAYER_MINIMAP := 1 << 1
const RENDER_LAYER_WORLD_AND_MINIMAP := RENDER_LAYER_WORLD | RENDER_LAYER_MINIMAP

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
@export var ground_noise_seed := 0  # 0 = a new ground layout on each start/reset
@export_range(0.005, 0.08, 0.001) var ground_noise_frequency := 0.022

enum Equip { SWORD, SPEAR, BOW, CROSSBOW, SHIELD }

# 装备位掩码（agent_equips）
const EQ_SWORD := 1 << Equip.SWORD
const EQ_SPEAR := 1 << Equip.SPEAR
const EQ_BOW := 1 << Equip.BOW
const EQ_CROSSBOW := 1 << Equip.CROSSBOW
const EQ_SHIELD := 1 << Equip.SHIELD

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

var agent_pos: PackedVector2Array  # _pick_agent 回读的当前位置
var spawn_x: PackedFloat32Array    # _spawn_agents 生成的出生位置（SoA，长度 MAX_AGENTS），供 agents.upload_agents
var spawn_y: PackedFloat32Array
const DISPLAY_LERP := 0.4

var world_size: Vector2
var goal_cells: PackedVector2Array
var _wall_sprite: Sprite2D
var _wall_tex_dirty := true
var _ground_sprite: Sprite2D
var _ground_tex_dirty := true

# GPU overlay system
var _ov_shd: RID
var _ov_pip: RID
var _ov_uset: RID
var _ov_buf_rgba: RID
var _ov_sprite: Sprite2D
var _ov_active_mode := -1
var _ov_raw := PackedByteArray()  # 异步回读到的 density / velocity 叠加 RGBA
# 选中 agent 所在 group 的阻挡格叠加（mode 2），独立缓冲 / uniform set / 纹理
var _blk_uset: RID
var _blk_buf_rgba: RID
var _blk_tex: CanvasTexture
var _blk_raw := PackedByteArray()  # 异步回读到的 RGBA

var swe_accum := 0.0
var swe_interval := 0.033

var show_density := false
var show_velocity := false
var paused := false

# Brush painting
enum BrushMode { NONE, WALL, ERASE, GROUND }
var brush_mode: BrushMode = BrushMode.NONE
var brush_radius := 2
var is_painting := false
var paint_erase := false
var terrain_dirty := false
var ground_dirty := false
var brush_ground_cost := 2.0
var _brush_label: Label

var perf_gpu := 0.0
var perf_read := 0.0
var perf_disppos := 0.0
var perf_mesh := 0.0
var perf_mm_readback_info := 0.0
var perf_mm_readback_dmg := 0.0
var perf_mm_loop := 0.0
var perf_mm_set_buffer := 0.0
var perf_alpha := 0.15
var perf_peak_total := 0.0
var perf_peak_detail := ""
var perf_render := 0.0
var _last_frame_usec := 0
var alive_count := 0
var selected_agent: int = -1
var nearest_enemy: int = -1
var sel_query := {}  # 选中 agent 的 GPU 查询结果（agents.take_select_inspect），sel == selected_agent 时有效
var select_label: Label

var multi_mesh: MultiMesh
var mm_instance: MultiMeshInstance2D
var mm_buf := PackedFloat32Array()

# Per-agent CPU data
var agent_equips: PackedByteArray  # 装备位掩码（EQ_*）
var agent_factions: PackedInt32Array
var cpu_agent_info: PackedInt32Array
var cpu_max_hp: PackedInt32Array
var cpu_regen_rate: PackedFloat32Array
var cpu_atk_range: PackedFloat32Array
var cpu_atk_damage: PackedFloat32Array
# 装备位掩码 → info 位 / 射程 / 伤害（_build_equip_tables）
var _equip_info: PackedInt32Array
var _equip_range: PackedFloat32Array
var _equip_damage: PackedFloat32Array

# Faction config
var alliance_masks: PackedInt32Array
var faction_to_group: PackedInt32Array
var faction_colors: Array[Color]
var group_goals: Array
var _goal_rr_idx := 0
var cached_info := PackedInt32Array()

var _agent_debounce_timer: Timer
var _agent_pending_count := -1

# GI (Radiance Cascades)
var gi: GPUGI
var _gi_sprite: Sprite2D
var _gi_enabled := true
var _gi_frame_skip := 2
var _gi_frame_counter := 0
var _gi_tex_w := 400
var _gi_tex_h := 240
var _gi_precision := 0.50
var _gi_emit_strength := GI_AGENT_EMIT_STRENGTH
var _laser_layers: Array[LaserOverlay] = []  # soft / glow / core
var _laser_count := 0  # 本帧 GPU 压缩出的激光数（可能超过 LASER_CAP）
var _laser_raw := PackedByteArray()  # 本帧到达的压缩激光整块（agents.take_frame_readback）
var _laser_points := PackedVector2Array()
var _laser_colors := PackedColorArray()

# HUD
var fps_label: Label
var hud_label: Label
var slider_agents: HSlider
var label_agents: Label
var slider_music_volume: HSlider
var label_music_volume: Label
var slider_gi_precision: HSlider
var label_gi_precision: Label
var btn_density: CheckButton
var btn_velocity: CheckButton
var btn_pause: Button
var bgm_player: AudioStreamPlayer
var _music_volume := 0.5
var cam: Camera2D
var cam_zoom := 1.0
var cam_dragging := false
var cam_drag_origin := Vector2.ZERO
var _gi_precision_debounce_timer: Timer
var _minimap_root: Control
var _minimap_texture: TextureRect
var _minimap_viewport: SubViewport
var _minimap_camera: Camera2D
var _minimap_overlay: MinimapOverlay


func _ready() -> void:
	world_size = Vector2(grid_width * cell_size, grid_height * cell_size)
	visibility_layer = RENDER_LAYER_WORLD_AND_MINIMAP

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
	_setup_goals()  # GPU BFS：需要 setup_bfs 之后、地形已上传

	_spawn_agents()
	agents.upload_agents(spawn_x, spawn_y, agent_count)
	_upload_combat()
	agents.upload_faction_colors(faction_colors)

	_setup_multimesh()
	_setup_overlay()
	_setup_laser_overlay()
	_setup_gi()
	_setup_hud()
	_play_bgm()


func _exit_tree() -> void:
	# 本地 RenderingDevice 要手动释放：先放各模块的 RID（GI、叠加层的 uniform set 引用 agents / field 的缓冲，先放），最后 free 设备本身
	if gi: gi.cleanup()
	_cleanup_overlay()
	if agents: agents.cleanup()
	if field: field.cleanup()
	if rd:
		rd.free()
		rd = null


func _setup_gi() -> void:
	if gi:
		gi.cleanup()

	var tex_size := _get_gi_texture_size()
	_gi_tex_w = tex_size.x
	_gi_tex_h = tex_size.y
	gi = GPUGI.new(rd, _gi_tex_w, _gi_tex_h)
	gi.laser_emit_strength = RANGED_LASER_GI_STRENGTH
	gi.laser_life = RANGED_LASER_LIFE
	_update_gi_buffers()

	if _gi_sprite == null:
		_gi_sprite = Sprite2D.new()
		_gi_sprite.centered = false
		_gi_sprite.z_index = 2
		_gi_sprite.visibility_layer = RENDER_LAYER_WORLD
		_gi_sprite.texture_filter = CanvasItem.TEXTURE_FILTER_LINEAR
		var mat := CanvasItemMaterial.new()
		mat.blend_mode = CanvasItemMaterial.BLEND_MODE_ADD
		_gi_sprite.material = mat
		_gi_sprite.modulate = Color(1.0, 1.0, 1.0, GI_DISPLAY_ALPHA)
		add_child(_gi_sprite)
	_gi_sprite.texture = null
	_gi_sprite.visible = _gi_enabled
	var gi_region := _get_gi_region_rect()
	_apply_gi_world_scale(gi_region)
	_update_gi_sprite_rect(gi_region)
	_update_gi_precision_label()


func _update_gi_buffers() -> void:
	if gi == null or agents == null:
		return
	gi.set_agent_buffers(
		agents.buf_pos_x[agents.cur],
		agents.buf_pos_y[agents.cur],
		agents.buf_agent_info,
		agents.buf_mm_fac_colors,
		agents.buf_hit_flash
	)
	gi.set_laser_buffer(agents.buf_laser_compact, GPUAgents.LASER_CAP)
	if field:
		gi.set_terrain_buffer(field.buf_terrain, grid_width, grid_height, cell_size)


func _setup_laser_overlay() -> void:
	if not _laser_layers.is_empty():
		return
	var mat := CanvasItemMaterial.new()
	mat.blend_mode = CanvasItemMaterial.BLEND_MODE_ADD
	# [宽度, alpha]：加法混合与绘制顺序无关，三层分开画等价于逐条交错画
	var layer_cfgs := [
		[RANGED_LASER_VIS_SOFT_WIDTH, 0.055],
		[RANGED_LASER_VIS_GLOW_WIDTH, 0.16],
		[RANGED_LASER_VIS_WIDTH, 0.62],
	]
	for cfg in layer_cfgs:
		var layer := LaserOverlay.new()
		layer.sim = self
		layer.width = cfg[0]
		layer.self_modulate = Color(1.0, 1.0, 1.0, cfg[1])
		layer.z_index = 4
		layer.visibility_layer = RENDER_LAYER_WORLD
		layer.material = mat
		layer.visible = false
		add_child(layer)
		_laser_layers.append(layer)


func _get_gi_texture_size() -> Vector2i:
	var vp_size := get_viewport().get_visible_rect().size
	return Vector2i(
		maxi(32, roundi(vp_size.x * _gi_precision)),
		maxi(32, roundi(vp_size.y * _gi_precision))
	)


func _get_gi_region_rect() -> Rect2:
	var vp_size := get_viewport().get_visible_rect().size
	var zoom := cam.zoom if cam else Vector2.ONE
	var view_size := Vector2(vp_size.x / zoom.x, vp_size.y / zoom.y)
	var region_size := view_size * GI_REGION_SCALE
	var region_pos := (cam.global_position if cam else world_size * 0.5) - region_size * 0.5
	return Rect2(region_pos, region_size)


func _update_gi_sprite_rect(region: Rect2) -> void:
	if _gi_sprite == null:
		return
	_gi_sprite.position = region.position
	_gi_sprite.scale = Vector2(region.size.x / float(_gi_tex_w), region.size.y / float(_gi_tex_h))


func _get_gi_world_per_texel(region: Rect2) -> float:
	var wx := region.size.x / maxf(float(_gi_tex_w), 1.0)
	var wy := region.size.y / maxf(float(_gi_tex_h), 1.0)
	return maxf((wx + wy) * 0.5, 0.001)


func _world_to_gi_texels(world_length: float, region: Rect2) -> float:
	return maxf(world_length / _get_gi_world_per_texel(region), 0.05)


func _get_gi_agent_splat_radius_world() -> float:
	return maxf(agent_radius * GI_AGENT_SPLAT_RADIUS_SCALE, cell_size * 0.45)


func _apply_gi_world_scale(region: Rect2) -> void:
	if gi == null:
		return
	var world_per_texel := _get_gi_world_per_texel(region)
	var reference_world_per_texel := GI_REGION_SCALE / GI_PRECISION_MIN
	gi.interval_scale = clampf(reference_world_per_texel / world_per_texel, 0.125, 8.0)
	gi.laser_width = _world_to_gi_texels(RANGED_LASER_GI_WIDTH_WORLD, region)


func _ensure_gi_texture_size() -> void:
	var wanted := _get_gi_texture_size()
	if wanted.x != _gi_tex_w or wanted.y != _gi_tex_h:
		_setup_gi()


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
	_generate_ground()

	_ground_sprite = Sprite2D.new()
	_ground_sprite.centered = false
	_ground_sprite.z_index = 0  # above the parent's background, below the later wall sprite
	_ground_sprite.visibility_layer = RENDER_LAYER_WORLD_AND_MINIMAP
	_ground_sprite.scale = Vector2(cell_size, cell_size)
	_ground_sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	var ground_shader := Shader.new()
	ground_shader.code = GROUND_SHADER_CODE
	var ground_mat := ShaderMaterial.new()
	ground_mat.shader = ground_shader
	_ground_sprite.material = ground_mat
	add_child(_ground_sprite)
	_ground_tex_dirty = true

	_wall_sprite = Sprite2D.new()
	_wall_sprite.centered = false
	_wall_sprite.z_index = 0
	_wall_sprite.visibility_layer = RENDER_LAYER_WORLD_AND_MINIMAP
	_wall_sprite.scale = Vector2(cell_size, cell_size)
	_wall_sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	var wall_shader := Shader.new()
	wall_shader.code = WALL_SHADER_CODE
	var wall_mat := ShaderMaterial.new()
	wall_mat.shader = wall_shader
	_wall_sprite.material = wall_mat
	add_child(_wall_sprite)
	_wall_tex_dirty = true


# 墙纹理是 terrain 本身（RF，每格一个纹素）：> 0.5 画墙色，其余透明
const WALL_SHADER_CODE := """
shader_type canvas_item;

void fragment() {
	COLOR = texture(TEXTURE, UV).r > 0.5 ? vec4(0.35, 0.35, 0.42, 1.0) : vec4(0.0);
}
"""


const GROUND_SHADER_CODE := """
shader_type canvas_item;

void fragment() {
	float cost = texture(TEXTURE, UV).r;
	float t = clamp((cost - 2.0) / 2.0, 0.0, 1.0);
	vec3 ground_color = mix(vec3(0.62, 0.44, 0.19), vec3(0.28, 0.18, 0.12), t);
	COLOR = vec4(ground_color, cost > 1.0 ? 0.8 : 0.0);
}
"""


func _generate_ground() -> void:
	var noise := FastNoiseLite.new()
	noise.seed = ground_noise_seed if ground_noise_seed != 0 else randi_range(1, 2147483647)
	noise.noise_type = FastNoiseLite.TYPE_SIMPLEX_SMOOTH
	noise.frequency = ground_noise_frequency
	noise.fractal_type = FastNoiseLite.FRACTAL_FBM
	noise.fractal_octaves = 3
	noise.fractal_gain = 0.5
	field.ground_cost.fill(1.0)
	# Broad normal areas, rough edges, then increasingly rare mud/deep-mud cores.
	# Sampling coherent noise keeps adjacent cells in natural patches.
	for y in range(1, grid_height - 1):
		for x in range(1, grid_width - 1):
			var i := field.idx(x, y)
			if field.terrain[i] > 0.5:
				continue
			var value := noise.get_noise_2d(x, y)
			if value > 0.50:
				field.ground_cost[i] = 4.0
			elif value > 0.26:
				field.ground_cost[i] = 3.0
			elif value > 0.04:
				field.ground_cost[i] = 2.0
	_ground_tex_dirty = true


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

	# 墙掩码（LA8，alpha = 255 为墙）：亮度 > 0.5 即 L8 >= 128。阈值逐像素，与旋转/翻转可交换，只算一次
	var lum := castle_base.get_data()
	var mask_data := PackedByteArray(); mask_data.resize(FOOT * FOOT * 2)
	if lum.size() >= FOOT * FOOT:
		for k in range(FOOT * FOOT):
			if lum[k] >= 128:
				mask_data[k * 2 + 1] = 255
	var mask_base := Image.create_from_data(FOOT, FOOT, false, Image.FORMAT_LA8, mask_data)
	# 地形整张作为 RF 图，每座城堡按掩码写入 1.0（墙）
	var wall_src := Image.create(FOOT, FOOT, false, Image.FORMAT_RF)
	wall_src.fill(Color(1.0, 0.0, 0.0))
	var terrain_img := Image.create_from_data(grid_width, grid_height, false, Image.FORMAT_RF, field.terrain.to_byte_array())

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

		var castle_img: Image = mask_base.duplicate()
		var rot := randi_range(0, 3)
		match rot:
			1: castle_img.rotate_90(CLOCKWISE)
			2: castle_img.rotate_180()
			3: castle_img.rotate_90(COUNTERCLOCKWISE)
		if randf() < 0.5:
			castle_img.flip_x()

		terrain_img.blit_rect_mask(wall_src, castle_img, Rect2i(0, 0, FOOT, FOOT), Vector2i(rx, ry))
		var walls := castle_img.get_data()  # LA8：alpha 在 (ly * FOOT + lx) * 2 + 1

		castle_centers.append(Vector2i(rx + FOOT / 2, ry + FOOT / 2))

		var best_exit := Vector2i(rx + FOOT / 2, ry)
		var best_dist := INF
		var center_to_map := Vector2(float(mcx - (rx + FOOT / 2)), float(mcy - (ry + FOOT / 2)))
		for lx in range(FOOT):
			for ly in [0, FOOT - 1]:
				if walls[(ly * FOOT + lx) * 2 + 1] == 0:
					var d := center_to_map.dot(Vector2(float(lx - FOOT / 2), float(ly - FOOT / 2)))
					if d > best_dist:
						continue
					best_dist = d
					best_exit = Vector2i(rx + lx, ry + ly)
		for ly in range(FOOT):
			for lx in [0, FOOT - 1]:
				if walls[(ly * FOOT + lx) * 2 + 1] == 0:
					var d := center_to_map.dot(Vector2(float(lx - FOOT / 2), float(ly - FOOT / 2)))
					if d > best_dist:
						continue
					best_dist = d
					best_exit = Vector2i(rx + lx, ry + ly)
		castle_exits.append(best_exit)

	field.terrain = terrain_img.get_data().to_float32_array()


func _setup_goals() -> void:
	goal_cells = PackedVector2Array()

	var center_goal: Array[Vector2i] = [Vector2i(grid_width / 2, grid_height / 2)]
	group_goals = []
	for _g in range(NUM_GOAL_GROUPS):
		group_goals.append(center_goal)
	field.build_all_goal_fields(group_goals)



# ── Agents ───────────────────────────────────────────────────────────────

func _spawn_agents() -> void:
	if _equip_info.is_empty():
		_build_equip_tables()
	var n := agent_count
	# 直接生成 GPU 要的 SoA 数组（长度 MAX_AGENTS，n 之后为 0）
	var px := PackedFloat32Array(); px.resize(MAX_AGENTS)
	var py := PackedFloat32Array(); py.resize(MAX_AGENTS)
	var info := PackedInt32Array(); info.resize(MAX_AGENTS)
	var rng := PackedFloat32Array(); rng.resize(MAX_AGENTS)
	var dmg := PackedFloat32Array(); dmg.resize(MAX_AGENTS)
	var equips := PackedByteArray(); equips.resize(n)
	var t_info := _equip_info
	var t_rng := _equip_range
	var t_dmg := _equip_damage
	var nc := castle_centers.size()
	var ccx := PackedFloat32Array(); ccx.resize(nc)
	var ccy := PackedFloat32Array(); ccy.resize(nc)
	for f in range(nc):
		ccx[f] = castle_centers[f].x
		ccy[f] = castle_centers[f].y
	var cs := cell_size
	var x_lo := cell_size * 3.0
	var x_hi := world_size.x - cell_size * 3.0
	var y_lo := cell_size * 3.0
	var y_hi := world_size.y - cell_size * 3.0

	# 随机数的调用顺序和次数与原来的逐 agent 写法相同（固定 seed 下结果不变）
	for i in range(n):
		var faction := i % NUM_FACTIONS
		if faction < nc:
			px[i] = (ccx[faction] + randf_range(-20.0, 20.0)) * cs
			py[i] = (ccy[faction] + randf_range(-20.0, 20.0)) * cs
		else:
			px[i] = randf_range(x_lo, x_hi)
			py[i] = randf_range(y_lo, y_hi)

		# Equipment（位掩码）
		var m := 0
		if randf() < 0.6:
			m = EQ_SWORD if randf() < 0.5 else EQ_SPEAR
		if randf() < 0.4:
			m |= EQ_BOW if randf() < 0.5 else EQ_CROSSBOW
		if m == 0:
			m = EQ_SWORD
		if randf() < 0.3:
			m |= EQ_SHIELD
		equips[i] = m
		info[i] = t_info[m] | ((faction & 0x1F) << 1)
		rng[i] = t_rng[m]
		dmg[i] = t_dmg[m]

	spawn_x = px
	spawn_y = py
	agent_equips = equips
	# agent i 的阵营为 i % NUM_FACTIONS：一个周期倍增拼接
	var fac := PackedInt32Array()
	for f in range(NUM_FACTIONS):
		fac.append(f)
	while fac.size() < n:
		fac.append_array(fac)
	fac.resize(n)
	agent_factions = fac
	cpu_agent_info = info
	cpu_max_hp = PackedInt32Array(); cpu_max_hp.resize(n); cpu_max_hp.fill(20000); cpu_max_hp.resize(MAX_AGENTS)
	cpu_regen_rate = PackedFloat32Array(); cpu_regen_rate.resize(n); cpu_regen_rate.fill(2.0); cpu_regen_rate.resize(MAX_AGENTS)
	cpu_atk_range = rng
	cpu_atk_damage = dmg


# 装备位掩码 → info 的存活/近战/远程/盾位、射程、伤害；累计规则同原来逐件遍历 EQUIP_DB
func _build_equip_tables() -> void:
	var count := 1 << Equip.size()
	_equip_info = PackedInt32Array(); _equip_info.resize(count)
	_equip_range = PackedFloat32Array(); _equip_range.resize(count)
	_equip_damage = PackedFloat32Array(); _equip_damage.resize(count)
	for m in range(count):
		var has_melee := false
		var has_ranged := false
		var has_shield := false
		var best_range := 0.0
		var best_dmg := 0.0
		for e in Equip.values():
			if (m & (1 << e)) == 0:
				continue
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
		if has_melee:  info |= (1 << 6)
		if has_ranged: info |= (1 << 7)
		if has_shield: info |= (1 << 8)
		_equip_info[m] = info
		_equip_range[m] = best_range
		_equip_damage[m] = best_dmg


func _upload_combat() -> void:
	agents.upload_combat_data(
		cpu_agent_info, cpu_max_hp,
		cpu_regen_rate, cpu_atk_range, cpu_atk_damage,
		alliance_masks, faction_to_group
	)


# ── Brush painting ───────────────────────────────────────────────────────

# terrain 直接作为 RF 纹理（墙色由 _wall_sprite 的 shader 决定）：首次新建，之后 update() 复用
func _rebuild_wall_texture() -> void:
	var img := Image.create_from_data(grid_width, grid_height, false, Image.FORMAT_RF, field.terrain.to_byte_array())
	var tex := _wall_sprite.texture as ImageTexture
	if tex == null:
		_wall_sprite.texture = ImageTexture.create_from_image(img)
	else:
		tex.update(img)
	_wall_tex_dirty = false


func _rebuild_ground_texture() -> void:
	var img := Image.create_from_data(grid_width, grid_height, false, Image.FORMAT_RF, field.ground_cost.to_byte_array())
	var tex := _ground_sprite.texture as ImageTexture
	if tex == null:
		_ground_sprite.texture = ImageTexture.create_from_image(img)
	else:
		tex.update(img)
	_ground_tex_dirty = false


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

			elif brush_mode == BrushMode.GROUND:
				if field.terrain[gi] > 0.5:
					continue
				var want := 1.0 if paint_erase else brush_ground_cost
				if field.ground_cost[gi] != want:
					field.ground_cost[gi] = want
					changed = true
					ground_dirty = true

			elif brush_mode == BrushMode.ERASE:
				if field.terrain[gi] > 0.5:
					field.terrain[gi] = 0.0
					changed = true
					terrain_dirty = true
				if field.ground_cost[gi] != 1.0:
					field.ground_cost[gi] = 1.0
					changed = true
					ground_dirty = true

	if changed:
		queue_redraw()


func _flush_paint() -> void:
	if terrain_dirty:
		field._upload(field.buf_terrain, field.terrain)
		_wall_tex_dirty = true
		terrain_dirty = false
	if ground_dirty:
		field._upload(field.buf_ground_cost, field.ground_cost)
		_ground_tex_dirty = true
		ground_dirty = false
	if _wall_tex_dirty:
		_rebuild_wall_texture()
	if _ground_tex_dirty:
		_rebuild_ground_texture()


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
	mm_instance.visibility_layer = RENDER_LAYER_WORLD_AND_MINIMAP
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

	var _ov_uniforms := func(out_buf: RID) -> Array[RDUniform]:
		return [
			_u.call(0, field.buf_density),
			_u.call(1, field.buf_terrain),
			_u.call(2, field.buf_goal_dist),
			_u.call(3, agents.buf_cell_blocked),
			_u.call(4, out_buf),
			_u.call(5, field.buf_out_vx),
			_u.call(6, field.buf_out_vy),
			_u.call(7, agents.buf_faction_presence),
		]

	_ov_uset = rd.uniform_set_create(_ov_uniforms.call(_ov_buf_rgba), _ov_shd, 0)

	_ov_sprite = Sprite2D.new()
	_ov_sprite.centered = false
	_ov_sprite.z_index = 1
	_ov_sprite.visibility_layer = RENDER_LAYER_WORLD
	_ov_sprite.scale = Vector2(cell_size, cell_size)
	_ov_sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	_ov_sprite.visible = false
	add_child(_ov_sprite)

	# 阻挡格叠加：同一 shader 的 mode 2，独立输出缓冲 / uniform set / 纹理，可与 density/velocity 同时显示。
	# 在 _draw 里原来逐格画的位置整张画出，层级不变（选中圈之下、agent 之下）；CanvasTexture 指定最近邻过滤
	_blk_buf_rgba = rd.storage_buffer_create(n_bytes, z)
	_blk_uset = rd.uniform_set_create(_ov_uniforms.call(_blk_buf_rgba), _ov_shd, 0)
	_blk_tex = CanvasTexture.new()
	_blk_tex.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST


func _dispatch_overlay(mode: int, param_i: int = 0, param_f: float = 0.0, group_off: int = 0, fac_mask: int = 0) -> void:
	var cl := rd.compute_list_begin()
	_record_overlay(cl, _ov_uset, mode, param_i, param_f, group_off, fac_mask)
	rd.compute_list_end()
	# 同一次提交里异步回读，数据在 sync() 里到达
	_ov_raw = PackedByteArray()
	rd.buffer_get_data_async(_ov_buf_rgba, _on_overlay_data, 0, grid_width * grid_height * 4)
	rd.submit()
	rd.sync()

	var raw := _ov_raw
	_ov_raw = PackedByteArray()
	if raw.size() != grid_width * grid_height * 4: return  # 数据缺失（理论上不会）：保留上一次的叠加
	_ov_sprite.texture = _overlay_texture(_ov_sprite.texture, raw)
	_ov_sprite.visible = true
	_ov_active_mode = mode


func _on_overlay_data(data: PackedByteArray) -> void:
	_ov_raw = data


# 在 cl 里记录一次 build_overlay.glsl，输出到 uset 绑定的缓冲（binding 4）
func _record_overlay(cl: int, uset: RID, mode: int, param_i: int = 0, param_f: float = 0.0, group_off: int = 0, fac_mask: int = 0) -> void:
	var p := PackedByteArray(); p.resize(32)
	p.encode_s32(0, grid_width)
	p.encode_s32(4, grid_height)
	p.encode_s32(8, mode)
	p.encode_s32(12, param_i)
	p.encode_float(16, param_f)
	p.encode_s32(20, group_off)
	p.encode_s32(24, fac_mask)

	rd.compute_list_bind_compute_pipeline(cl, _ov_pip)
	rd.compute_list_bind_uniform_set(cl, uset, 0)
	rd.compute_list_set_push_constant(cl, p, 32)
	rd.compute_list_dispatch(cl, ceili(float(grid_width) / 8.0), ceili(float(grid_height) / 8.0), 1)


# 用叠加的 RGBA 数据更新纹理：已有 ImageTexture 就 update() 复用，只在第一次新建
func _overlay_texture(tex: ImageTexture, raw: PackedByteArray) -> ImageTexture:
	var img := Image.create_from_data(grid_width, grid_height, false, Image.FORMAT_RGBA8, raw)
	if tex == null:
		return ImageTexture.create_from_image(img)
	tex.update(img)
	return tex


func _hide_overlay() -> void:
	if _ov_sprite:
		_ov_sprite.visible = false
	_ov_active_mode = -1


func _cleanup_overlay() -> void:
	if rd == null:
		return
	# uniform set 先放（依赖 _ov_shd 和各缓冲）；已被 Godot 连带释放的跳过
	for us in [_blk_uset, _ov_uset]:
		if us.is_valid() and rd.uniform_set_is_valid(us): rd.free_rid(us)
	for rid in [_blk_buf_rgba, _ov_pip, _ov_shd, _ov_buf_rgba]:
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


# 用随主提交 sync() 到达的数据（agents.request_frame_readback）更新 MultiMesh 和计数
func _sync_multimesh_gpu() -> void:
	var ta := Time.get_ticks_usec()
	var frame := agents.take_frame_readback(agent_count)
	if frame.is_empty(): return  # 数据缺失（理论上不会）：保留上一帧的显示
	var mm_raw: PackedByteArray = frame.mm
	mm_buf = mm_raw.to_float32_array()
	alive_count = frame.alive
	_laser_count = frame.laser_count
	_laser_raw = frame.lasers
	var tb := Time.get_ticks_usec()
	multi_mesh.set_buffer(mm_buf)
	var tc := Time.get_ticks_usec()
	var alpha := perf_alpha
	perf_mm_readback_info = lerpf(perf_mm_readback_info, float(tb - ta) / 1000.0, alpha)
	perf_mm_readback_dmg  = 0.0
	perf_mm_loop          = 0.0
	perf_mm_set_buffer    = lerpf(perf_mm_set_buffer, float(tc - tb) / 1000.0, alpha)


# 重置 / 改数量后：在 GPU 上按刚上传的出生位置构建一次 MultiMesh，随这次提交回读（暂停时画面也立即正确）
func _rebuild_multimesh() -> void:
	agents.reset_alive_counter()
	var cl := rd.compute_list_begin()
	agents.dispatch_build_mm(cl, 1.0, true)
	rd.compute_list_end()
	agents.request_frame_readback(agent_count)
	rd.submit()
	rd.sync()
	_sync_multimesh_gpu()


func _sync_laser_overlay() -> void:
	if _laser_layers.is_empty():
		return
	var active := _laser_count > 0
	if active:
		# 从到达的激光整块切出前 n 条：点区 [0, n×16)，颜色区从 LASER_CAP×16 起 n×16 字节
		var n := mini(_laser_count, GPUAgents.LASER_CAP)
		var col_off := GPUAgents.LASER_CAP * 16
		_laser_points = _laser_raw.slice(0, n * 16).to_vector2_array()
		_laser_colors = _laser_raw.slice(col_off, col_off + n * 16).to_color_array()
	else:
		_laser_points = PackedVector2Array()
		_laser_colors = PackedColorArray()
	for layer in _laser_layers:
		layer.visible = active
		if active:
			layer.queue_redraw()


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
	_refresh_selection()
	_update_select_label()
	queue_redraw()


# 选中改变后立刻查询一次（暂停时也能正确显示）；之后每帧由 _process 在主 compute list 里查询
func _refresh_selection() -> void:
	sel_query = {}
	if selected_agent < 0 or selected_agent >= agent_count:
		return
	var cl := rd.compute_list_begin()
	_record_select(cl, true)
	rd.compute_list_end()
	_request_select_readback()
	rd.submit()
	rd.sync()
	_apply_select_readback()


# 在 cl 里记录选中查询（select_inspect 的 4 个小 pass）+ 阻挡格叠加（build_overlay mode 2）
func _record_select(cl: int, after_swap: bool) -> void:
	agents.dispatch_select_inspect(cl, selected_agent, after_swap)
	# 阵营位在 GPU 上不会改变；记录时本次查询结果还没出来，直接用 CPU 侧的阵营
	var sel_grp := faction_to_group[agent_factions[selected_agent]]
	_record_overlay(cl, _blk_uset, 2, 1 << sel_grp)


# compute_list_end 之后、submit 之前：登记查询结果（几十字节）和阻挡格叠加的异步回读，
# 数据在随后的 sync() 里到达，不额外 flush + stall
func _request_select_readback() -> void:
	agents.request_select_inspect()
	_blk_raw = PackedByteArray()
	rd.buffer_get_data_async(_blk_buf_rgba, _on_block_overlay_data)


func _on_block_overlay_data(data: PackedByteArray) -> void:
	_blk_raw = data


# sync() 之后：取出查询结果，更新阻挡格叠加纹理
func _apply_select_readback() -> void:
	sel_query = agents.take_select_inspect(selected_agent)
	if _blk_raw.size() == grid_width * grid_height * 4:
		_blk_tex.diffuse_texture = _overlay_texture(_blk_tex.diffuse_texture, _blk_raw)
	_blk_raw = PackedByteArray()


func _update_select_label() -> void:
	if select_label == null:
		return
	if selected_agent < 0 or selected_agent >= agent_count or sel_query.get("sel", -1) != selected_agent:
		select_label.text = ""
		return
	var idx := selected_agent
	var info: int = sel_query.info
	var alive := (info & 1) != 0
	var fac := (info >> 1) & 0x1F
	var is_atk := (info & (1 << 9)) != 0
	var team := "G%d" % fac
	var dmg: int = sel_query.dmg
	var max_hp_val := cpu_max_hp[idx] if idx < cpu_max_hp.size() else 20000
	var hp_cur := maxf(float(max_hp_val - dmg) / 100.0, 0.0)
	var hp_max := float(max_hp_val) / 100.0

	var equip_names: PackedStringArray = PackedStringArray()
	if idx < agent_equips.size():
		var em := agent_equips[idx]
		for e in Equip.values():
			if (em & (1 << e)) == 0:
				continue
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
	var pos_v: Vector2 = sel_query.pos

	# Nearest enemy（GPU 查询：alliance 之外的存活 agent 中距离²最小者，平局取最小下标）
	nearest_enemy = sel_query.ne
	var ne_str := ""
	if nearest_enemy >= 0:
		var ne_dist := sqrt(float(sel_query.ne_dsq))
		var ne_fac := (int(sel_query.ne_info) >> 1) & 0x1F
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
	_queue_minimap_redraw()

	# WASD camera pan
	if cam:
		var move := Vector2.ZERO
		if Input.is_key_pressed(KEY_W): move.y -= 1.0
		if Input.is_key_pressed(KEY_S): move.y += 1.0
		if Input.is_key_pressed(KEY_A): move.x -= 1.0
		if Input.is_key_pressed(KEY_D): move.x += 1.0
		if move != Vector2.ZERO:
			cam.position += move.normalized() * CAM_PAN_SPEED * dt / cam_zoom

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
	agents.dispatch_combat(cl, dt, engage_range, attack_cd_base, RANGED_LASER_LIFE,
						   HIT_GI_FLASH_DECAY_PER_SEC, HIT_GI_FLASH_VALUE)
	rd.compute_list_add_barrier(cl)

	# 5.5 — Cell blocked map (per-group blockage bitmask)
	agents.dispatch_cell_blocked(cl)
	rd.compute_list_add_barrier(cl)

	# 6 — Agent steer + integrate
	agents.dispatch_steer(cl, dt, separation_radius, separation_strength,
						  steering_responsiveness, world_size.x, world_size.y)
	rd.compute_list_add_barrier(cl)

	# 7 — Build MultiMesh buffer on GPU
	agents.dispatch_build_mm(cl, DISPLAY_LERP)
	rd.compute_list_add_barrier(cl)

	# 7.5 — 远程激光压缩（供批量绘制）
	agents.dispatch_laser_compact(cl, RANGED_LASER_LIFE)

	# 7.6 — 选中 agent：GPU 查询（读 steer 本帧写出的位置，上面的 barrier 已保证 steer 完成）+ 阻挡格叠加
	var sel_active := selected_agent >= 0 and selected_agent < agent_count
	if sel_active:
		_record_select(cl, false)

	# 7.7 — Dynamic goal update (GPU BFS, 1 group per frame round-robin)
	# 只读本帧 density 写的 faction_presence；写出的 goal_dist_all/gdir 下一帧才被 velocity/combat 读
	field.dispatch_goal_bfs(cl, _goal_rr_idx, NUM_FACTIONS)
	_goal_rr_idx = (_goal_rr_idx + 1) % NUM_GOAL_GROUPS

	rd.compute_list_end()
	# 本帧 mm / 计数 / 激光的异步回读，数据随下面的 sync() 到达（t0→t1 含这部分拷贝）
	agents.request_frame_readback(agent_count)
	if sel_active:
		_request_select_readback()
	rd.submit()
	rd.sync()
	var t1 := Time.get_ticks_usec()

	# 8 — Swap double-buffer; 选中时取出随 sync 到达的查询结果 / 阻挡格叠加
	agents.swap_buffers()
	var t1b := Time.get_ticks_usec()
	if sel_active:
		_apply_select_readback()
	var t1c := Time.get_ticks_usec()
	_sync_multimesh_gpu()
	_sync_laser_overlay()
	var t2 := Time.get_ticks_usec()
	_update_select_label()

	# 8.5 — GI (Radiance Cascades) — every N frames
	if _gi_enabled and _gi_sprite != null:
		_gi_frame_counter += 1
		if _gi_frame_counter >= _gi_frame_skip:
			_gi_frame_counter = 0
			_ensure_gi_texture_size()
			_update_gi_buffers()
			var gi_region := _get_gi_region_rect()
			_apply_gi_world_scale(gi_region)
			_update_gi_sprite_rect(gi_region)
			var splat_radius := _world_to_gi_texels(_get_gi_agent_splat_radius_world(), gi_region)
			# 激光读本帧主提交压缩出的紧凑列表（_sync_multimesh_gpu 刚取到的条数）
			gi.compute_gi(agent_count, gi_region.position, gi_region.size,
						   splat_radius, _gi_emit_strength, _laser_count)
			var gi_img := gi.get_output_image()
			if _gi_sprite.texture == null:
				_gi_sprite.texture = ImageTexture.create_from_image(gi_img)
				gi_img.save_png("user://gi_debug_stage%d.png" % gi.debug_stage)
				print("[GI] Saved debug image: user://gi_debug_stage%d.png" % gi.debug_stage)
				# Also save viewport screenshot
				await get_tree().process_frame
				get_viewport().get_texture().get_image().save_png("user://gi_viewport.png")
				print("[GI] Saved viewport: user://gi_viewport.png")
			else:
				(_gi_sprite.texture as ImageTexture).update(gi_img)

	var t3 := Time.get_ticks_usec()

	# 10 — GPU overlay (density / velocity), filtered by selected agent's faction
	var ov_group_off := 0
	var ov_fac_mask := 0
	if selected_agent >= 0 and sel_query.get("sel", -1) == selected_agent:
		var sel_info: int = sel_query.info
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
		perf_peak_detail = "GPU=%.1f Rd=%.1f MM=%.1f" % [
			gpu_ms, rd_ms, mm_ms]
	if frame_total > 50.0:
		print("[SPIKE] %.1f ms | GPU=%.1f Rd=%.1f MM=%.1f" % [
			frame_total, gpu_ms, rd_ms, mm_ms])
	fps_label.text = "FPS: %d" % Engine.get_frames_per_second()
	hud_label.text = (
		"GPU %.1f  Rd %.1f  MM %.1f | Alive: %d / %d\n" % [perf_gpu, perf_read, perf_mesh, alive_count, agent_count]
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
	player.volume_db = _music_volume_to_db(_music_volume)
	player.bus = "Master"
	add_child(player)
	bgm_player = player
	player.play()
	player.finished.connect(player.play)


func _music_volume_to_db(value: float) -> float:
	var linear := clampf(value, 0.0, 1.0)
	if linear <= 0.001:
		return -80.0
	return linear_to_db(linear)


# ── HUD ──────────────────────────────────────────────────────────────────

func _setup_hud() -> void:
	_agent_debounce_timer = Timer.new()
	_agent_debounce_timer.one_shot = true
	_agent_debounce_timer.wait_time = 1.0
	_agent_debounce_timer.timeout.connect(_apply_agent_count)
	add_child(_agent_debounce_timer)

	_gi_precision_debounce_timer = Timer.new()
	_gi_precision_debounce_timer.one_shot = true
	_gi_precision_debounce_timer.wait_time = 0.25
	_gi_precision_debounce_timer.timeout.connect(_apply_gi_precision)
	add_child(_gi_precision_debounce_timer)

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
	label_music_volume = Label.new()
	label_music_volume.add_theme_font_size_override("font_size", 12)
	label_music_volume.add_theme_color_override("font_color", Color(0.9, 0.75, 0.4))
	vbox.add_child(label_music_volume)
	slider_music_volume = _slider(0, 100, _music_volume * 100.0, 1)
	slider_music_volume.value_changed.connect(_on_music_volume_changed)
	vbox.add_child(slider_music_volume)
	_update_music_volume_label()

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
	label_gi_precision = Label.new()
	label_gi_precision.add_theme_font_size_override("font_size", 12)
	label_gi_precision.add_theme_color_override("font_color", Color(0.9, 0.75, 0.4))
	vbox.add_child(label_gi_precision)
	slider_gi_precision = _slider(GI_PRECISION_MIN, GI_PRECISION_MAX, _gi_precision, GI_PRECISION_STEP)
	slider_gi_precision.value_changed.connect(_on_gi_precision_changed)
	vbox.add_child(slider_gi_precision)
	_update_gi_precision_label()

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
	_brush_label = brush_lbl
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
	var ground_button := Button.new()
	ground_button.text = "[4] Ground: x%d | Speed: %d%%" % [int(brush_ground_cost), int(100.0 / brush_ground_cost)]
	ground_button.tooltip_text = "Paint ground with LMB; RMB restores normal ground.\nDarker ground is harder to cross. [3] Erase clears walls and ground."
	ground_button.pressed.connect(func(): _set_brush(BrushMode.GROUND))
	vbox.add_child(ground_button)
	var slider_ground := _slider(1, GPUSWEField.GROUND_COST_MAX, brush_ground_cost, 1)
	slider_ground.tooltip_text = "Ground difficulty: 1 = normal, 2 = rough, 3 = mud, 4 = deep mud."
	slider_ground.value_changed.connect(func(v: float):
		brush_ground_cost = v
		ground_button.text = "[4] Ground: x%d | Speed: %d%%" % [int(v), int(100.0 / v)]
		_set_brush(BrushMode.GROUND)
	)
	vbox.add_child(slider_ground)

	#_setup_minimap(canvas)


func _setup_minimap(canvas: CanvasLayer) -> void:
	_minimap_viewport = SubViewport.new()
	_minimap_viewport.disable_3d = true
	_minimap_viewport.transparent_bg = false
	_minimap_viewport.canvas_cull_mask = RENDER_LAYER_MINIMAP
	_minimap_viewport.render_target_update_mode = SubViewport.UPDATE_ALWAYS
	_minimap_viewport.world_2d = get_viewport().world_2d
	canvas.add_child(_minimap_viewport)

	_minimap_camera = Camera2D.new()
	_minimap_camera.position = world_size * 0.5
	_minimap_camera.enabled = true
	_minimap_viewport.add_child(_minimap_camera)
	_minimap_camera.make_current()

	_minimap_root = Control.new()
	_minimap_root.name = "Minimap"
	_minimap_root.mouse_filter = Control.MOUSE_FILTER_STOP
	_minimap_root.clip_contents = true
	canvas.add_child(_minimap_root)

	_minimap_texture = TextureRect.new()
	_minimap_texture.texture = _minimap_viewport.get_texture()
	_minimap_texture.stretch_mode = TextureRect.STRETCH_SCALE
	_minimap_texture.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_minimap_root.add_child(_minimap_texture)

	_minimap_overlay = MinimapOverlay.new()
	_minimap_overlay.sim = self
	_minimap_overlay.mouse_filter = Control.MOUSE_FILTER_IGNORE
	_minimap_root.add_child(_minimap_overlay)

	_update_minimap_layout()
	get_viewport().size_changed.connect(_update_minimap_layout)


func _update_minimap_layout() -> void:
	if _minimap_root == null or _minimap_viewport == null:
		return
	var vp_size := get_viewport().get_visible_rect().size
	var minimap_size := Vector2(
		maxf(1.0, vp_size.x * MINIMAP_SCREEN_FRACTION),
		maxf(1.0, vp_size.y * MINIMAP_SCREEN_FRACTION)
	)
	_minimap_root.position = Vector2(MINIMAP_MARGIN, vp_size.y - minimap_size.y - MINIMAP_MARGIN)
	_minimap_root.size = minimap_size
	_minimap_texture.position = Vector2.ZERO
	_minimap_texture.size = minimap_size
	_minimap_overlay.position = Vector2.ZERO
	_minimap_overlay.size = minimap_size
	_minimap_viewport.size = Vector2i(
		maxi(1, roundi(minimap_size.x)),
		maxi(1, roundi(minimap_size.y))
	)

	if _minimap_camera:
		_minimap_camera.position = world_size * 0.5
		var fit_zoom := minf(minimap_size.x / world_size.x, minimap_size.y / world_size.y)
		_minimap_camera.zoom = Vector2(fit_zoom, fit_zoom)
	_queue_minimap_redraw()


func _queue_minimap_redraw() -> void:
	if _minimap_overlay:
		_minimap_overlay.queue_redraw()


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
		BrushMode.GROUND: return "Ground x%d" % int(brush_ground_cost)
		_: return "None"

func _update_agent_label() -> void:
	label_agents.text = "Agents: %d" % agent_count
	label_agents.add_theme_color_override("font_color", Color(0.9, 0.75, 0.4))


func _update_music_volume_label() -> void:
	if label_music_volume == null:
		return
	label_music_volume.text = "Music Volume: %d%%" % roundi(_music_volume * 100.0)


func _on_music_volume_changed(val: float) -> void:
	_music_volume = clampf(val / 100.0, 0.0, 1.0)
	if bgm_player:
		bgm_player.volume_db = _music_volume_to_db(_music_volume)
	_update_music_volume_label()


func _update_gi_precision_label() -> void:
	if label_gi_precision == null:
		return
	var tex_size := _get_gi_texture_size()
	label_gi_precision.text = "GI Precision: %.2fx  (%d x %d)" % [_gi_precision, tex_size.x, tex_size.y]


func _on_gi_precision_changed(val: float) -> void:
	_gi_precision = snappedf(val, GI_PRECISION_STEP)
	_update_gi_precision_label()
	_gi_precision_debounce_timer.start()


func _apply_gi_precision() -> void:
	_setup_gi()
	_update_gi_precision_label()


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
	_spawn_agents()
	agents.upload_agents(spawn_x, spawn_y, agent_count)
	_upload_combat()
	agents.clear_corpse_map()
	agents.clear_laser_events()
	_laser_count = 0
	agents.clear_hit_flash_events()
	_rebuild_multimesh()
	_sync_laser_overlay()

func _toggle_pause() -> void:
	paused = not paused
	btn_pause.text = "  Resume  " if paused else "  Pause  "
	queue_redraw()

func _reset_sim() -> void:
	selected_agent = -1
	nearest_enemy = -1
	field.reset_flux()
	field.terrain.fill(0.0)
	ground_dirty = false
	terrain_dirty = false
	_ground_tex_dirty = true
	field.add_wall(0, 0, grid_width, 1)
	field.add_wall(0, grid_height - 1, grid_width, grid_height)
	field.add_wall(0, 0, 1, grid_height)
	field.add_wall(grid_width - 1, 0, grid_width, grid_height)
	_gen_castles()
	_generate_ground()
	_wall_tex_dirty = true
	_spawn_agents()
	agents.upload_agents(spawn_x, spawn_y, agent_count)
	_upload_combat()
	agents.clear_corpse_map()
	agents.clear_laser_events()
	_laser_count = 0
	agents.clear_hit_flash_events()
	field.upload_static_data()
	_setup_goals()  # GPU BFS 读地形，放在地形上传之后
	_rebuild_multimesh()
	_sync_laser_overlay()


# ── Drawing ──────────────────────────────────────────────────────────────

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, world_size), Color(0.11, 0.11, 0.14))
	var cs_val := cell_size

	for gc in goal_cells:
		draw_rect(Rect2(gc.x * cs_val, gc.y * cs_val, cs_val, cs_val), Color(0.2, 0.75, 0.35, 0.18))

	# 选中信息全部来自 GPU 查询（sel_query）
	if selected_agent >= 0 and sel_query.get("sel", -1) == selected_agent:
		var sel_pos: Vector2 = sel_query.pos
		var sel_info: int = sel_query.info
		var sel_alive := (sel_info & 1) != 0
		var sel_col := Color(1.0, 1.0, 0.2, 0.8) if sel_alive else Color(1.0, 0.3, 0.3, 0.6)

		# 选中 agent 所在 group 的阻挡格：GPU 叠加纹理一次画出（每格一个纹素）
		if _blk_tex.diffuse_texture != null:
			draw_texture_rect(_blk_tex, Rect2(Vector2.ZERO, world_size), false)

		draw_arc(sel_pos, agent_radius * 2.5, 0, TAU, 32, sel_col, 2.0)
		draw_arc(sel_pos, agent_radius * 3.5, 0, TAU, 32, Color(sel_col.r, sel_col.g, sel_col.b, 0.3), 1.0)
		if selected_agent < cpu_atk_range.size():
			var rng := cpu_atk_range[selected_agent]
			if rng > 0.0:
				draw_arc(sel_pos, rng, 0, TAU, 64, Color(1.0, 0.4, 0.2, 0.5), 1.5)
		if int(sel_query.ne) >= 0:
			var ne_pos: Vector2 = sel_query.ne_pos
			draw_line(sel_pos, ne_pos, Color(1.0, 0.2, 0.2, 0.7), 1.5)
			draw_arc(ne_pos, agent_radius * 2.0, 0, TAU, 24, Color(1.0, 0.2, 0.2, 0.8), 2.0)

		var sel_is_disp := (sel_info & (1 << 10)) != 0
		if sel_is_disp:
			var ci: int = sel_query.cell
			var sel_cx := ci % grid_width
			var sel_cy := ci / grid_width
			var blocker: int = sel_query.blocker
			draw_rect(Rect2(sel_cx * cs_val, sel_cy * cs_val, cs_val, cs_val),
					  Color(1.0, 0.5, 0.0, 0.5))
			if blocker >= 0 and blocker != selected_agent and blocker < agent_count:
				var bpos: Vector2 = sel_query.blk_pos
				draw_line(sel_pos, bpos, Color(1.0, 0.6, 0.0, 0.8), 2.0)
				draw_arc(bpos, agent_radius * 2.0, 0, TAU, 24, Color(1.0, 0.6, 0.0, 0.9), 2.0)

	if brush_mode != BrushMode.NONE:
		var mpos := get_global_mouse_position()
		var col: Color
		match brush_mode:
			BrushMode.WALL:  col = Color(0.4, 0.8, 1.0, 0.35)
			BrushMode.ERASE: col = Color(1.0, 0.4, 0.4, 0.35)
			BrushMode.GROUND: col = Color(0.85, 0.6, 0.25, 0.55)
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
			KEY_V:
				show_velocity = not show_velocity
				btn_velocity.button_pressed = show_velocity
				if not show_velocity and not show_density: _hide_overlay()
			KEY_L:
				_gi_enabled = not _gi_enabled
				if _gi_sprite:
					_gi_sprite.visible = _gi_enabled
				# 关闭期间的 history 已过时，重新打开时不再与它混合
				if _gi_enabled and gi: gi.temporal_valid = false
			KEY_F12:
				var img := get_viewport().get_texture().get_image()
				var path := "user://screenshot_%d.png" % Time.get_ticks_msec()
				img.save_png(path)
				print("[Screenshot] Saved: %s" % path)
			KEY_1: _set_brush(BrushMode.NONE)
			KEY_2: _set_brush(BrushMode.WALL)
			KEY_3: _set_brush(BrushMode.ERASE)
			KEY_4: _set_brush(BrushMode.GROUND)

	if brush_mode == BrushMode.NONE:
		if event is InputEventMouseButton:
			if event.pressed:
				if event.button_index == MOUSE_BUTTON_LEFT:
					_pick_agent(get_global_mouse_position())
				elif event.button_index == MOUSE_BUTTON_RIGHT:
					selected_agent = -1
					nearest_enemy = -1
					_refresh_selection()
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
	is_painting = false
	if _brush_label:
		_brush_label.text = "Mode: %s | Radius: %d" % [_brush_name(), brush_radius]
	queue_redraw()
