extends SceneTree

# Run with Vulkan (not --headless): godot --path . --script res://tests/ground_cost_test.gd
const Field = preload("res://scripts/gpu_swe_field.gd")
const Sim = preload("res://scripts/crowd_sim.gd")
const W := 35
const H := 21
const N := W * H
const INF := 0xFFFFFFFE
const WALL := 0xFFFFFFFF
var failures := 0
var rd: RenderingDevice
var field: GPUSWEField
var presence: RID
var factions: RID


func _initialize() -> void:
	root.hide()
	call_deferred("_run")


func _expect(ok: bool, message: String) -> void:
	if not ok:
		failures += 1
		push_error(message)


func _run() -> void:
	rd = RenderingServer.create_local_rendering_device()
	if rd == null:
		push_error("Vulkan RenderingDevice required; run without --headless")
		quit(1)
		return
	field = Field.new(rd, W, H, 8.0, 2)
	var fp := PackedInt32Array(); fp.resize(N)
	presence = rd.storage_buffer_create(N * 4, fp.to_byte_array())
	var ft := PackedInt32Array(); ft.resize(32); ft[1] = 1
	factions = rd.storage_buffer_create(128, ft.to_byte_array())
	field.setup_bfs(presence, factions)
	var seed_cell := Vector2i(W - 3, H / 2)
	var seed_idx := field.idx(seed_cell.x, seed_cell.y)

	# Flat ground is the old BFS, across multiple tiles and partial edge tiles.
	_border()
	_verify_fields("flat", seed_cell, seed_idx)

	# A costly rectangle should make the cheapest route go around its edge.
	for y in range(3, H - 3):
		for x in range(12, 23):
			field.ground_cost[field.idx(x, y)] = 4.0
	_verify_fields("detour", seed_cell, seed_idx)
	var costs := rd.buffer_get_data(field.buf_bfs_cost).to_int32_array()
	var at := field.idx(3, H / 2)
	var entered_mud := false
	for _step in range(N):
		if at == seed_idx:
			break
		var next := at
		for nb in _neighbors(at):
			if costs[nb] >= 0 and costs[nb] < costs[next]:
				next = nb
		if next == at:
			break
		at = next
		entered_mud = entered_mud or field.ground_cost[at] > 1.0
	_expect(at == seed_idx and not entered_mud, "cheapest route should avoid the mud rectangle")
	var gx := rd.buffer_get_data(field.buf_gdir_x, 0, N * 4).to_float32_array()
	var gy := rd.buffer_get_data(field.buf_gdir_y, 0, N * 4).to_float32_array()
	at = field.idx(11, H / 2)
	_expect(absf(gy[at]) > 0.1, "gradient must choose a detour at the symmetric mud boundary")
	_expect(gx[at] <= 0.0, "gradient must not steer straight into the expensive rectangle")

	# Force traversal: costly ground is passable, physical attack distance stays unchanged.
	field.terrain.fill(1.0)
	for x in range(1, W - 1):
		field.terrain[field.idx(x, H / 2)] = 0.0
	_verify_fields("mud corridor", seed_cell, seed_idx)

	# Inaccessible regions must keep sentinels, even next to a cost-4 cell.
	_border()
	field.add_wall(W / 2, 0, W / 2 + 1, H)
	field.ground_cost.fill(4.0)
	_verify_fields("disconnected", seed_cell, seed_idx)

	# Independent Dijkstra oracle catches weighted tile propagation and cost increases/decreases.
	var rng := RandomNumberGenerator.new(); rng.seed = 20261010
	for trial in range(4):
		_border()
		for y in range(1, H - 1):
			for x in range(1, W - 1):
				var i := field.idx(x, y)
				field.ground_cost[i] = float(rng.randi_range(1, 4))
				if rng.randf() < 0.15:
					field.terrain[i] = 1.0
		field.terrain[seed_idx] = 0.0
		_verify_fields("random %d" % trial, seed_cell, seed_idx)

	fp.fill(0)
	rd.buffer_update(presence, 0, N * 4, fp.to_byte_array())
	_dispatch_goal()
	var dirs := rd.buffer_get_data(field.buf_gdir_x, 0, N * 4).to_float32_array()
	_expect(dirs.count(0.0) == N, "no enemies must yield zero movement direction")
	_test_velocity()
	_test_brush()
	field.cleanup()
	rd.free_rid(presence)
	rd.free_rid(factions)
	rd.free()
	print("GROUND TESTS: %s (%d failures)" % ["PASS" if failures == 0 else "FAIL", failures])
	quit(0 if failures == 0 else 1)


func _border() -> void:
	field.terrain.fill(0.0)
	field.ground_cost.fill(1.0)
	field.add_wall(0, 0, W, 1)
	field.add_wall(0, H - 1, W, H)
	field.add_wall(0, 0, 1, H)
	field.add_wall(W - 1, 0, W, H)


func _neighbors(i: int) -> Array[int]:
	var result: Array[int] = []
	if i % W > 0: result.append(i - 1)
	if i % W < W - 1: result.append(i + 1)
	if i >= W: result.append(i - W)
	if i < N - W: result.append(i + W)
	return result


func _dijkstra(seed_idx: int, weighted: bool) -> PackedInt64Array:
	var dist := PackedInt64Array(); dist.resize(N); dist.fill(INF)
	var done := PackedByteArray(); done.resize(N)
	for i in range(N):
		if field.terrain[i] > 0.5:
			dist[i] = WALL
	dist[seed_idx] = 0
	for _step in range(N):
		var best := -1
		var best_dist := INF
		for i in range(N):
			if done[i] == 0 and dist[i] < best_dist:
				best = i
				best_dist = dist[i]
		if best < 0:
			break
		done[best] = 1
		for nb in _neighbors(best):
			if field.terrain[nb] > 0.5:
				continue
			var step_cost := int(field.ground_cost[nb]) if weighted else 1
			dist[nb] = mini(dist[nb], best_dist + step_cost)
	return dist


func _verify_fields(label: String, seed_cell: Vector2i, seed_idx: int) -> void:
	field.upload_static_data()
	var expected_cost := _dijkstra(seed_idx, true)
	var expected_dist := _dijkstra(seed_idx, false)
	field.build_all_goal_fields([[seed_cell]])
	_compare_distances(label + " startup", expected_cost, expected_dist)
	var startup := rd.buffer_get_data(field.buf_goal_dist_all).to_float32_array()
	for i in range(N):
		_expect(startup[i] == startup[N + i], label + " startup groups mismatch at %d" % i)
	var fp := PackedInt32Array(); fp.resize(N); fp[seed_idx] = 1 << 1
	rd.buffer_update(presence, 0, N * 4, fp.to_byte_array())
	_dispatch_goal()
	_compare_distances(label + " dynamic", expected_cost, expected_dist)
	print("PASS field oracle: ", label)


func _dispatch_goal() -> void:
	var cl := rd.compute_list_begin()
	field.dispatch_goal_bfs(cl, 0, 2)
	rd.compute_list_end()
	rd.submit()
	rd.sync()


func _compare_distances(label: String, expected_cost: PackedInt64Array, expected_dist: PackedInt64Array) -> void:
	var raw_cost := rd.buffer_get_data(field.buf_bfs_cost)
	var raw_dist := rd.buffer_get_data(field.buf_bfs_dist)
	var combat := rd.buffer_get_data(field.buf_goal_dist_all, 0, N * 4).to_float32_array()
	var gx := rd.buffer_get_data(field.buf_gdir_x, 0, N * 4).to_float32_array()
	var gy := rd.buffer_get_data(field.buf_gdir_y, 0, N * 4).to_float32_array()
	for i in range(N):
		if raw_cost.decode_u32(i * 4) != expected_cost[i] or raw_dist.decode_u32(i * 4) != expected_dist[i]:
			_expect(false, "%s at %d: cost %d/%d dist %d/%d" % [label, i, raw_cost.decode_u32(i * 4), expected_cost[i], raw_dist.decode_u32(i * 4), expected_dist[i]])
			return
		if expected_dist[i] < INF:
			_expect(combat[i] == float(expected_dist[i]), label + " changed combat range at %d" % i)
		if expected_cost[i] < INF and expected_cost[i] != expected_dist[i]:
			_expect(Vector2(gx[i], gy[i]).length() > 0.99, label + " weighted gradient stalled at %d" % i)
			if gx[i] < -0.001: _expect(i % W > 0 and expected_cost[i - 1] < expected_cost[i], label + " uphill left")
			if gx[i] > 0.001: _expect(i % W < W - 1 and expected_cost[i + 1] < expected_cost[i], label + " uphill right")
			if gy[i] < -0.001: _expect(i >= W and expected_cost[i - W] < expected_cost[i], label + " uphill up")
			if gy[i] > 0.001: _expect(i < N - W and expected_cost[i + W] < expected_cost[i], label + " uphill down")


func _test_velocity() -> void:
	_border()
	var direction := PackedFloat32Array(); direction.resize(N * 2); direction.fill(1.0)
	field._upload(field.buf_gdir_x, direction)
	direction.fill(0.0)
	field._upload(field.buf_gdir_y, direction)
	for i in range(N):
		field.ground_cost[i] = float(i % 4 + 1)
	field.upload_static_data()
	for density in [0.0, 2.0]:
		var values := PackedFloat32Array(); values.resize(N); values.fill(density)
		field._upload(field.buf_density, values)
		var cl := rd.compute_list_begin()
		field.dispatch_velocity(cl, 2)
		rd.compute_list_end()
		rd.submit()
		rd.sync()
		var vx := rd.buffer_get_data(field.buf_out_vx).to_float32_array()
		for g in range(2):
			for i in range(N):
				var expected: float = 0.0 if field.terrain[i] > 0.5 else field.agent_speed / (1.0 + density * field.density_slowdown) / field.ground_cost[i]
				if absf(vx[g * N + i] - expected) > 0.0001:
					_expect(false, "velocity density=%s group=%d cell=%d: %s/%s" % [density, g, i, vx[g * N + i], expected])
					break
	print("PASS velocity: all four costs, idle/occupied tiles, two groups, wall zero speed")


func _test_brush() -> void:
	_border()
	var sim := Sim.new()
	sim.field = field
	sim.grid_width = W
	sim.grid_height = H
	sim.cell_size = 8.0
	sim.brush_radius = 1
	sim.brush_mode = Sim.BrushMode.GROUND
	sim.brush_ground_cost = 4.0
	var center := Vector2(10.5, 10.5) * sim.cell_size
	var i := field.idx(10, 10)
	sim._apply_brush(center)
	_expect(field.ground_cost[i] == 4.0 and field.terrain[i] == 0.0 and sim.ground_dirty, "ground paint must set cost without blocking")
	sim.paint_erase = true
	sim._apply_brush(center)
	_expect(field.ground_cost[i] == 1.0, "right mouse must restore normal ground")
	field.terrain[i] = 1.0
	sim.paint_erase = false
	sim._apply_brush(center)
	_expect(field.ground_cost[i] == 1.0 and field.terrain[i] == 1.0, "ground paint must skip walls")
	field.ground_cost[i] = 3.0
	sim.brush_mode = Sim.BrushMode.ERASE
	sim._apply_brush(center)
	_expect(field.terrain[i] == 0.0 and field.ground_cost[i] == 1.0, "erase must clear wall and ground cost")
	sim.brush_mode = Sim.BrushMode.GROUND
	sim._apply_brush(Vector2(0.5, 0.5) * sim.cell_size)
	_expect(field.terrain[0] == 1.0 and field.ground_cost[0] == 1.0, "brush must preserve world boundary")
	sim.field = null
	sim.free()
	print("PASS brush: paint, restore, erase, wall/boundary protection")
