extends Node2D
## FluidCrowd — fully GPU-driven agent simulation.
##
## Pipeline per frame (single compute list):
##   1. GPUAgents builds counting-sort cell buffer
##   2. GPUAgents computes bilinear density from cell buffer
##   3. GPUSWEField runs SWE flux + velocity on the density
##   4. GPUAgents steers agents using velocity field + cell buffer
##   5. CPU readback positions → MultiMesh

const MAX_AGENTS := 3000

@export var grid_width := 80
@export var grid_height := 50
@export var cell_size := 12.0
@export var agent_count := 500
@export var agent_radius := 3.5
@export var separation_radius := 18.0
@export var separation_strength := 120.0
@export var steering_responsiveness := 6.0

var rd: RenderingDevice
var field: GPUSWEField
var agents: GPUAgents

var agent_pos: PackedVector2Array
var agent_vel: PackedVector2Array

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
var perf_mesh := 0.0
var perf_alpha := 0.15

var multi_mesh: MultiMesh
var mm_instance: MultiMeshInstance2D

var hud_label: Label
var slider_agents: HSlider
var label_agents: Label
var slider_speed: HSlider
var slider_density_scale: HSlider
var slider_separation: HSlider
var btn_density: CheckButton
var btn_velocity: CheckButton
var btn_goal: CheckButton
var btn_pause: Button


func _ready() -> void:
	world_size = Vector2(grid_width * cell_size, grid_height * cell_size)

	rd = RenderingServer.create_local_rendering_device()
	assert(rd != null, "Vulkan RenderingDevice required")

	field = GPUSWEField.new(rd, grid_width, grid_height, cell_size)
	_build_environment()
	field.upload_static_data()

	agents = GPUAgents.new(rd, MAX_AGENTS, separation_radius, world_size)
	agents.set_field_buffers(
		field.buf_out_vx, field.buf_out_vy,
		field.buf_terrain, field.buf_goal_dist,
		field.buf_density,
		grid_width, grid_height, cell_size
	)
	agents.build_uniform_sets()

	_spawn_agents()
	agents.upload_agents(agent_pos, agent_vel, agent_count)

	_setup_multimesh()
	_setup_hud()


func _exit_tree() -> void:
	if agents:
		agents.cleanup()
	if field:
		field.cleanup()


# ── Environment ──────────────────────────────────────────────────────────

func _build_environment() -> void:
	field.add_wall(0, 0, grid_width, 1)
	field.add_wall(0, grid_height - 1, grid_width, grid_height)
	field.add_wall(0, 0, 1, grid_height)
	field.add_wall(grid_width - 1, 0, grid_width, grid_height)

	var wx := grid_width / 2 - 1
	var gap_lo := grid_height / 2 - 4
	var gap_hi := grid_height / 2 + 4
	field.add_wall(wx, 1, wx + 3, gap_lo)
	field.add_wall(wx, gap_hi, wx + 3, grid_height - 1)

	var px := int(grid_width * 0.7)
	field.add_wall(px, grid_height / 2 - 8, px + 2, grid_height / 2 - 2)
	field.add_wall(px, grid_height / 2 + 2, px + 2, grid_height / 2 + 8)

	_update_wall_cells()

	goal_mask = PackedByteArray()
	goal_mask.resize(grid_width * grid_height)
	for y in range(grid_height / 2 - 6, grid_height / 2 + 6):
		for x in range(grid_width - 4, grid_width - 1):
			goal_mask[y * grid_width + x] = 1
	_rebuild_goals()


# ── Agents ───────────────────────────────────────────────────────────────

func _spawn_agents() -> void:
	agent_pos = PackedVector2Array(); agent_pos.resize(agent_count)
	agent_vel = PackedVector2Array(); agent_vel.resize(agent_count)
	for i in range(agent_count):
		agent_pos[i] = Vector2(
			randf_range(2.0 * cell_size, 12.0 * cell_size),
			randf_range(3.0 * cell_size, (grid_height - 3.0) * cell_size)
		)
		agent_vel[i] = Vector2.ZERO


# ── Brush painting ───────────────────────────────────────────────────────

func _update_wall_cells() -> void:
	wall_cells = PackedVector2Array()
	for y in range(grid_height):
		for x in range(grid_width):
			if field.terrain[field.idx(x, y)] > 0.5:
				wall_cells.append(Vector2(x, y))


func _rebuild_goals() -> void:
	var goals: Array[Vector2i] = []
	goal_cells = PackedVector2Array()
	for y in range(grid_height):
		for x in range(grid_width):
			if goal_mask[y * grid_width + x] > 0:
				goals.append(Vector2i(x, y))
				goal_cells.append(Vector2(x, y))
	if goals.is_empty():
		goals.append(Vector2i(grid_width - 2, grid_height / 2))
		goal_cells.append(Vector2(grid_width - 2, grid_height / 2))
		goal_mask[(grid_height / 2) * grid_width + grid_width - 2] = 1
	field.build_goal_field(goals)


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
	multi_mesh.instance_count = MAX_AGENTS
	multi_mesh.visible_instance_count = agent_count
	multi_mesh.mesh = quad
	mm_instance = MultiMeshInstance2D.new()
	mm_instance.multimesh = multi_mesh
	mm_instance.texture = _make_circle_texture(16)
	mm_instance.modulate = Color(0.92, 0.58, 0.2)
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


func _sync_multimesh() -> void:
	for i in range(agent_count):
		multi_mesh.set_instance_transform_2d(i, Transform2D(0.0, agent_pos[i]))


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

	# 2 — Density from cell buffer → field.buf_density
	agents.dispatch_density(cl)
	rd.compute_list_add_barrier(cl)

	# 3 — SWE flux (throttled)
	swe_accum += dt
	if swe_accum >= swe_interval:
		field.dispatch_swe(cl, swe_accum)
		swe_accum = 0.0
		rd.compute_list_add_barrier(cl)

	# 4 — Velocity field
	field.dispatch_velocity(cl)
	rd.compute_list_add_barrier(cl)

	# 5 — Agent steer + integrate
	agents.dispatch_steer(cl, dt, separation_radius, separation_strength,
						  steering_responsiveness, world_size.x, world_size.y)

	rd.compute_list_end()
	rd.submit()
	rd.sync()
	var t1 := Time.get_ticks_usec()

	# 6 — Readback
	agent_pos = agents.readback_positions()
	var t2 := Time.get_ticks_usec()

	# 7 — MultiMesh
	_sync_multimesh()
	var t3 := Time.get_ticks_usec()

	# Optional overlay readback
	if show_density:
		field.readback_density()
	if show_velocity:
		field.readback_velocity()

	var a := perf_alpha
	perf_gpu  = lerpf(perf_gpu,  float(t1 - t0) / 1000.0, a)
	perf_read = lerpf(perf_read, float(t2 - t1) / 1000.0, a)
	perf_mesh = lerpf(perf_mesh, float(t3 - t2) / 1000.0, a)
	var total := perf_gpu + perf_read + perf_mesh
	hud_label.text = (
		"Total: %.2f ms | FPS: %d\n" % [total, Engine.get_frames_per_second()]
		+ "  GPU pipeline %.2f ms\n" % perf_gpu
		+ "  Readback     %.2f ms\n" % perf_read
		+ "  MultiMesh    %.2f ms" % perf_mesh
	)

	if show_density or show_velocity or show_goal or brush_mode != BrushMode.NONE:
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

	var panel := PanelContainer.new()
	panel.position = Vector2(1280 - 260, 40)
	panel.custom_minimum_size = Vector2(250, 0)
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
	title.text = "FluidCrowd  [Full GPU]"
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
	s.custom_minimum_size.x = 220
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
	_update_agent_label()
	multi_mesh.visible_instance_count = agent_count
	_spawn_agents()
	agents.upload_agents(agent_pos, agent_vel, agent_count)
	_sync_multimesh()

func _toggle_pause() -> void:
	paused = not paused
	btn_pause.text = "  Resume  " if paused else "  Pause  "
	queue_redraw()

func _reset_sim() -> void:
	field.reset_flux()
	_spawn_agents()
	agents.upload_agents(agent_pos, agent_vel, agent_count)
	_sync_multimesh()


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

	# Brush cursor
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
			KEY_1: _set_brush(BrushMode.NONE)
			KEY_2: _set_brush(BrushMode.WALL)
			KEY_3: _set_brush(BrushMode.GOAL)
			KEY_4: _set_brush(BrushMode.ERASE)

	if brush_mode == BrushMode.NONE:
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
