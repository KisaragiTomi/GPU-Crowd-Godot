extends Node2D
## TaskSim — GPU-driven task planning demo.
##
## Pipeline per frame:
##   1. CPU: clear task claim buffer, maybe respawn done tasks
##   2. GPU: task_score  → task_assign → task_navigate → task_complete
##   3. CPU: readback positions, stamina, targets, task status
##   4. CPU: render agents (MultiMesh), tasks, assignment lines

const MAX_AGENTS := 500
const MAX_TASKS  := 300

@export var grid_width := 80
@export var grid_height := 50
@export var cell_size := 12.0
@export var agent_count := 30
@export var task_count := 60

var rd: RenderingDevice
var planner: GPUTaskPlanner
var world_size: Vector2

# CPU-side agent state (from readback)
var agent_pos: PackedVector2Array
var agent_vel: PackedVector2Array
var agent_stamina_cpu: PackedFloat32Array
var agent_targets_cpu: PackedInt32Array

# CPU-side task state
var task_pos_cpu: PackedVector2Array
var task_cost_cpu: PackedFloat32Array
var task_type_cpu: PackedInt32Array
var task_status_cpu: PackedInt32Array

# Terrain
var terrain: PackedFloat32Array

var paused := false
var show_lines := true

# Timing
var perf_gpu := 0.0
var perf_read := 0.0
var perf_alpha := 0.15

# Task respawn
var respawn_timer := 0.0
const RESPAWN_INTERVAL := 2.5

# BFS flow-field slot management
const MAX_SLOTS := 64
var slot_to_task: Array[int]
var task_to_slot: PackedInt32Array
var slot_free: Array[int]

# Rendering
var multi_mesh: MultiMesh
var mm_instance: MultiMeshInstance2D
var hud_label: Label
var btn_pause: Button
var slider_agents: HSlider
var label_agents: Label
var slider_speed: HSlider
var slider_decay: HSlider
var slider_sep: HSlider
var btn_lines: CheckButton


func _ready() -> void:
	world_size = Vector2(grid_width * cell_size, grid_height * cell_size)

	rd = RenderingServer.create_local_rendering_device()
	assert(rd != null, "Vulkan RenderingDevice required")

	planner = GPUTaskPlanner.new(rd, MAX_AGENTS, MAX_TASKS,
								 grid_width, grid_height, cell_size)

	_build_terrain()
	planner.upload_terrain(terrain)

	_spawn_agents()
	_spawn_tasks()
	_upload_all()
	planner.build_uniform_sets()
	_init_bfs_slots()

	_setup_multimesh()
	_setup_hud()


func _exit_tree() -> void:
	if planner:
		planner.cleanup()


# ── Terrain ──────────────────────────────────────────────────────────────

func _build_terrain() -> void:
	terrain = PackedFloat32Array()
	terrain.resize(grid_width * grid_height)

	for x in range(grid_width):
		terrain[x] = 1.0
		terrain[(grid_height - 1) * grid_width + x] = 1.0
	for y in range(grid_height):
		terrain[y * grid_width] = 1.0
		terrain[y * grid_width + grid_width - 1] = 1.0

	var bx := grid_width / 3
	for y in range(grid_height / 4, grid_height / 4 + 8):
		for x in range(bx, bx + 2):
			terrain[y * grid_width + x] = 1.0

	var bx2 := int(grid_width * 0.6)
	for y in range(int(grid_height * 0.55), int(grid_height * 0.55) + 10):
		for x in range(bx2, bx2 + 2):
			terrain[y * grid_width + x] = 1.0


func _is_wall(gx: int, gy: int) -> bool:
	if gx < 0 or gx >= grid_width or gy < 0 or gy >= grid_height:
		return true
	return terrain[gy * grid_width + gx] > 0.5


func _random_open_pos() -> Vector2:
	for _attempt in range(200):
		var gx := randi_range(2, grid_width - 3)
		var gy := randi_range(2, grid_height - 3)
		if not _is_wall(gx, gy):
			return Vector2((float(gx) + 0.5) * cell_size,
						   (float(gy) + 0.5) * cell_size)
	return Vector2(cell_size * 4.0, cell_size * 4.0)


# ── Agents ───────────────────────────────────────────────────────────────

func _spawn_agents() -> void:
	agent_pos = PackedVector2Array(); agent_pos.resize(agent_count)
	agent_vel = PackedVector2Array(); agent_vel.resize(agent_count)
	agent_stamina_cpu = PackedFloat32Array(); agent_stamina_cpu.resize(agent_count)
	agent_targets_cpu = PackedInt32Array(); agent_targets_cpu.resize(agent_count)

	for i in range(agent_count):
		agent_pos[i] = _random_open_pos()
		agent_vel[i] = Vector2.ZERO
		agent_stamina_cpu[i] = randf_range(0.7, 1.0)
		agent_targets_cpu[i] = -1


# ── Tasks ────────────────────────────────────────────────────────────────

func _spawn_tasks() -> void:
	task_pos_cpu = PackedVector2Array(); task_pos_cpu.resize(task_count)
	task_cost_cpu = PackedFloat32Array(); task_cost_cpu.resize(task_count)
	task_type_cpu = PackedInt32Array(); task_type_cpu.resize(task_count)
	task_status_cpu = PackedInt32Array(); task_status_cpu.resize(task_count)

	var rest_count := int(task_count * 0.2)
	for i in range(task_count):
		task_pos_cpu[i] = _random_open_pos()
		task_status_cpu[i] = 0
		if i < rest_count:
			task_type_cpu[i] = 1
			task_cost_cpu[i] = randf_range(-0.40, -0.25)
		else:
			task_type_cpu[i] = 0
			task_cost_cpu[i] = randf_range(0.06, 0.14)


func _upload_all() -> void:
	var stam := PackedFloat32Array(); stam.resize(MAX_AGENTS)
	for i in range(agent_count):
		stam[i] = agent_stamina_cpu[i]
	planner.upload_agents(agent_pos, agent_vel, stam, agent_count)

	var tpx := PackedFloat32Array(); tpx.resize(MAX_TASKS)
	var tpy := PackedFloat32Array(); tpy.resize(MAX_TASKS)
	var tc  := PackedFloat32Array(); tc.resize(MAX_TASKS)
	var tp  := PackedFloat32Array(); tp.resize(MAX_TASKS)
	var tt  := PackedInt32Array();   tt.resize(MAX_TASKS)
	var ts  := PackedInt32Array();   ts.resize(MAX_TASKS)
	for i in range(task_count):
		tpx[i] = task_pos_cpu[i].x
		tpy[i] = task_pos_cpu[i].y
		tc[i]  = task_cost_cpu[i]
		tp[i]  = 1.0
		tt[i]  = task_type_cpu[i]
		ts[i]  = task_status_cpu[i]
	planner.upload_tasks(tpx, tpy, tc, tp, tt, ts, task_count)


func _maybe_respawn_tasks(dt: float) -> void:
	respawn_timer += dt
	if respawn_timer < RESPAWN_INTERVAL:
		return
	respawn_timer = 0.0

	var done_idx := PackedInt32Array()
	var new_px := PackedFloat32Array()
	var new_py := PackedFloat32Array()
	var new_cost := PackedFloat32Array()
	var new_pri := PackedFloat32Array()
	var new_type := PackedInt32Array()

	var rest_count := int(task_count * 0.2)
	for i in range(task_count):
		if task_status_cpu[i] == 3:
			var pos := _random_open_pos()
			done_idx.append(i)
			new_px.append(pos.x)
			new_py.append(pos.y)
			new_pri.append(1.0)
			if i < rest_count:
				new_type.append(1)
				new_cost.append(randf_range(-0.40, -0.25))
			else:
				new_type.append(0)
				new_cost.append(randf_range(0.06, 0.14))

	if done_idx.size() > 0:
		planner.update_tasks_partial(done_idx, new_px, new_py,
									 new_cost, new_pri, new_type)
		for k in range(done_idx.size()):
			var idx := done_idx[k]
			task_pos_cpu[idx] = Vector2(new_px[k], new_py[k])
			task_cost_cpu[idx] = new_cost[k]
			task_type_cpu[idx] = new_type[k]
			task_status_cpu[idx] = 0
			_invalidate_task_slot(idx)


# ── BFS flow fields ─────────────────────────────────────────────────────

func _init_bfs_slots() -> void:
	slot_to_task = []
	slot_to_task.resize(MAX_SLOTS)
	for k in range(MAX_SLOTS):
		slot_to_task[k] = -1
	task_to_slot = PackedInt32Array()
	task_to_slot.resize(MAX_TASKS)
	task_to_slot.fill(-1)
	slot_free = []
	for k in range(MAX_SLOTS - 1, -1, -1):
		slot_free.append(k)
	planner.upload_task_slot_map(task_to_slot)


func _alloc_slot() -> int:
	if slot_free.is_empty():
		return -1
	return slot_free.pop_back()


func _free_slot(slot: int) -> void:
	if slot < 0 or slot >= MAX_SLOTS:
		return
	var old_task := slot_to_task[slot]
	if old_task >= 0 and old_task < task_to_slot.size():
		task_to_slot[old_task] = -1
	slot_to_task[slot] = -1
	slot_free.append(slot)


func _invalidate_task_slot(task_idx: int) -> void:
	if task_idx < 0 or task_idx >= task_to_slot.size():
		return
	var slot := task_to_slot[task_idx]
	if slot >= 0:
		_free_slot(slot)


func _bfs_from(gx: int, gy: int) -> PackedFloat32Array:
	var cc := grid_width * grid_height
	var dist := PackedFloat32Array()
	dist.resize(cc)
	dist.fill(1e6)
	if gx < 0 or gx >= grid_width or gy < 0 or gy >= grid_height:
		return dist
	if terrain[gy * grid_width + gx] > 0.5:
		return dist
	var start := gy * grid_width + gx
	dist[start] = 0.0
	var q: Array[int] = [start]
	var head := 0
	var DX: Array[int] = [1, -1, 0, 0]
	var DY: Array[int] = [0, 0, 1, -1]
	while head < q.size():
		var ci := q[head]; head += 1
		var cx := ci % grid_width
		var cy := ci / grid_width
		var cd := dist[ci]
		for d in range(4):
			var nx := cx + DX[d]
			var ny := cy + DY[d]
			if nx < 0 or nx >= grid_width or ny < 0 or ny >= grid_height:
				continue
			var ni := ny * grid_width + nx
			if terrain[ni] > 0.5:
				continue
			if cd + 1.0 < dist[ni]:
				dist[ni] = cd + 1.0
				q.append(ni)
	return dist


func _update_flow_fields() -> void:
	var active_tasks := {}
	for i in range(agent_count):
		var tgt: int = agent_targets_cpu[i] if i < agent_targets_cpu.size() else -1
		if tgt >= 0 and tgt < task_count:
			active_tasks[tgt] = true

	for slot in range(MAX_SLOTS):
		var t := slot_to_task[slot]
		if t >= 0 and not active_tasks.has(t):
			_free_slot(slot)

	var dirty := false
	for t_idx in active_tasks:
		if task_to_slot[t_idx] >= 0:
			continue
		var slot := _alloc_slot()
		if slot < 0:
			break
		slot_to_task[slot] = t_idx
		task_to_slot[t_idx] = slot
		var pos := task_pos_cpu[t_idx]
		var gx := int(pos.x / cell_size)
		var gy := int(pos.y / cell_size)
		var dist_field := _bfs_from(gx, gy)
		planner.upload_goal_dist_slot(slot, dist_field)
		dirty = true

	if dirty or active_tasks.size() > 0:
		planner.upload_task_slot_map(task_to_slot)


# ── MultiMesh ────────────────────────────────────────────────────────────

func _setup_multimesh() -> void:
	var quad := QuadMesh.new()
	quad.size = Vector2(7.0, 7.0)
	multi_mesh = MultiMesh.new()
	multi_mesh.transform_format = MultiMesh.TRANSFORM_2D
	multi_mesh.use_colors = true
	multi_mesh.instance_count = MAX_AGENTS
	multi_mesh.visible_instance_count = agent_count
	multi_mesh.mesh = quad
	mm_instance = MultiMeshInstance2D.new()
	mm_instance.multimesh = multi_mesh
	mm_instance.texture = _make_circle_texture(8)
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


func _stamina_color(s: float) -> Color:
	if s > 0.6:
		return Color(0.2, 0.85, 0.3)
	elif s > 0.28:
		var t := (s - 0.28) / 0.32
		return Color(1.0, 0.95, 0.2).lerp(Color(0.2, 0.85, 0.3), t)
	elif s > 0.14:
		var t := (s - 0.14) / 0.14
		return Color(1.0, 0.5, 0.15).lerp(Color(1.0, 0.95, 0.2), t)
	else:
		return Color(0.95, 0.2, 0.15)


func _sync_multimesh() -> void:
	for i in range(agent_count):
		multi_mesh.set_instance_transform_2d(i, Transform2D(0.0, agent_pos[i]))
		var s := agent_stamina_cpu[i] if i < agent_stamina_cpu.size() else 1.0
		multi_mesh.set_instance_color(i, _stamina_color(s))


# ── Simulation loop ─────────────────────────────────────────────────────

func _physics_process(dt: float) -> void:
	if paused:
		return

	_maybe_respawn_tasks(dt)
	_update_flow_fields()
	planner.clear_claims()

	var t0 := Time.get_ticks_usec()

	var cl := rd.compute_list_begin()

	planner.dispatch_score(cl)
	rd.compute_list_add_barrier(cl)

	planner.dispatch_assign(cl)
	rd.compute_list_add_barrier(cl)

	planner.dispatch_navigate(cl, dt)
	rd.compute_list_add_barrier(cl)

	planner.dispatch_complete(cl, dt)

	rd.compute_list_end()
	rd.submit()
	rd.sync()

	var t1 := Time.get_ticks_usec()

	agent_pos = planner.readback_positions()
	agent_stamina_cpu = planner.readback_stamina()
	agent_targets_cpu = planner.readback_targets()
	task_status_cpu = planner.readback_task_status()

	var t2 := Time.get_ticks_usec()

	_sync_multimesh()

	var a := perf_alpha
	perf_gpu  = lerpf(perf_gpu,  float(t1 - t0) / 1000.0, a)
	perf_read = lerpf(perf_read, float(t2 - t1) / 1000.0, a)

	var avail := 0; var claimed := 0; var done := 0
	for i in range(task_count):
		match task_status_cpu[i]:
			0: avail += 1
			1: claimed += 1
			3: done += 1

	var avg_stam := 0.0
	for i in range(agent_count):
		avg_stam += agent_stamina_cpu[i]
	avg_stam /= maxf(float(agent_count), 1.0)

	var total := perf_gpu + perf_read
	hud_label.text = (
		"GPU: %.2f ms  Read: %.2f ms  FPS: %d\n" % [perf_gpu, perf_read, Engine.get_frames_per_second()]
		+ "Agents: %d   Avg stamina: %.0f%%\n" % [agent_count, avg_stam * 100.0]
		+ "Tasks:  avail %d  claimed %d  done %d" % [avail, claimed, done]
	)

	queue_redraw()


# ── Drawing ──────────────────────────────────────────────────────────────

func _draw() -> void:
	draw_rect(Rect2(Vector2.ZERO, world_size), Color(0.09, 0.09, 0.12))
	var cs := cell_size

	for y in range(grid_height):
		for x in range(grid_width):
			if terrain[y * grid_width + x] > 0.5:
				draw_rect(Rect2(x * cs, y * cs, cs, cs), Color(0.3, 0.3, 0.38))

	for i in range(task_count):
		var st := task_status_cpu[i] if i < task_status_cpu.size() else 0
		if st == 3:
			continue
		var tp := task_pos_cpu[i]
		var tt := task_type_cpu[i]
		var sz := 5.0
		var col: Color
		if tt >= 1:
			col = Color(0.2, 0.8, 0.35, 0.85) if st == 0 else Color(0.3, 1.0, 0.5, 0.95)
		else:
			col = Color(0.3, 0.5, 0.95, 0.75) if st == 0 else Color(0.5, 0.7, 1.0, 0.9)
		draw_rect(Rect2(tp.x - sz, tp.y - sz, sz * 2, sz * 2), col)
		if st == 1:
			draw_rect(Rect2(tp.x - sz - 1, tp.y - sz - 1, sz * 2 + 2, sz * 2 + 2),
					  Color(col.r, col.g, col.b, 0.5), false, 1.5)

	if show_lines:
		for i in range(agent_count):
			var tgt: int = agent_targets_cpu[i] if i < agent_targets_cpu.size() else -1
			if tgt >= 0 and tgt < task_count:
				var tc := task_type_cpu[tgt]
				var lc: Color
				if tc >= 1:
					lc = Color(0.3, 0.9, 0.4, 0.25)
				else:
					lc = Color(0.5, 0.65, 1.0, 0.2)
				draw_line(agent_pos[i], task_pos_cpu[tgt], lc, 1.0)


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
	title.text = "TaskPlanner  [GPU]"
	title.add_theme_font_size_override("font_size", 16)
	title.add_theme_color_override("font_color", Color(0.4, 0.75, 0.95))
	title.horizontal_alignment = HORIZONTAL_ALIGNMENT_CENTER
	vbox.add_child(title)

	vbox.add_child(_sep())

	label_agents = Label.new()
	label_agents.add_theme_font_size_override("font_size", 12)
	vbox.add_child(label_agents)
	slider_agents = _slider(5, MAX_AGENTS, agent_count, 5)
	slider_agents.value_changed.connect(_on_agents_changed)
	vbox.add_child(slider_agents)
	_update_agent_label()

	vbox.add_child(_sep())

	vbox.add_child(_lbl("Nav Speed"))
	slider_speed = _slider(20, 200, planner.nav_speed, 5)
	slider_speed.value_changed.connect(func(v: float): planner.nav_speed = v)
	vbox.add_child(slider_speed)

	vbox.add_child(_lbl("Stamina Decay / sec"))
	slider_decay = _slider(0.0, 0.08, planner.stamina_decay, 0.005)
	slider_decay.value_changed.connect(func(v: float): planner.stamina_decay = v)
	vbox.add_child(slider_decay)

	vbox.add_child(_lbl("Separation Force"))
	slider_sep = _slider(0, 300, planner.sep_strength, 10)
	slider_sep.value_changed.connect(func(v: float): planner.sep_strength = v)
	vbox.add_child(slider_sep)

	vbox.add_child(_sep())

	btn_lines = _toggle("Assignment lines", show_lines)
	btn_lines.toggled.connect(func(on: bool): show_lines = on; queue_redraw())
	vbox.add_child(btn_lines)

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
	vbox.add_child(_lbl("[Space] Pause  [R] Reset"))


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

func _update_agent_label() -> void:
	label_agents.text = "Agents: %d" % agent_count
	label_agents.add_theme_color_override("font_color", Color(0.9, 0.75, 0.4))

func _on_agents_changed(val: float) -> void:
	agent_count = int(val)
	_update_agent_label()
	multi_mesh.visible_instance_count = agent_count
	_spawn_agents()
	_spawn_tasks()
	_upload_all()
	_init_bfs_slots()

func _toggle_pause() -> void:
	paused = not paused
	btn_pause.text = "  Resume  " if paused else "  Pause  "

func _reset_sim() -> void:
	_spawn_agents()
	_spawn_tasks()
	_upload_all()
	_init_bfs_slots()
	respawn_timer = 0.0


# ── Input ────────────────────────────────────────────────────────────────

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventKey and event.pressed:
		match event.keycode:
			KEY_SPACE: _toggle_pause()
			KEY_R: _reset_sim()
			KEY_L:
				show_lines = not show_lines
				btn_lines.button_pressed = show_lines
				queue_redraw()
