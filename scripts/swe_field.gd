class_name SWEField
extends RefCounted
## Shallow Water Equation crowd velocity field.
##
## Models crowd density as water height on a 2D grid.
## Virtual-pipe flux method routes flow toward goals (low effective height)
## while diverting around obstacles (high terrain) and congestion (high density).
##
## Effective height per cell:  H = density * ds + goal_dist * gs + terrain * ws
## Flux flows from high H → low H, producing a velocity field that agents follow.

var gw: int
var gh: int
var cs: float

# Grid arrays — flat row-major, index = y * gw + x
var terrain: PackedFloat32Array
var goal_dist: PackedFloat32Array
var density: PackedFloat32Array

# SWE virtual-pipe flux (persistent across frames)
var fr: PackedFloat32Array
var fl: PackedFloat32Array
var fu: PackedFloat32Array
var fd: PackedFloat32Array

# Pre-computed goal direction (unit vectors from BFS gradient)
var gdir_x: PackedFloat32Array
var gdir_y: PackedFloat32Array

# Output velocity
var out_vx: PackedFloat32Array
var out_vy: PackedFloat32Array

# Tuning
var gravity := 25.0
var damping := 0.90
var density_scale := 2.0
var goal_scale := 1.0
var wall_scale := 500.0
var max_flux := 50.0
var agent_speed := 70.0
var density_slowdown := 0.3


func _init(width: int, height: int, cell_size: float) -> void:
	gw = width
	gh = height
	cs = cell_size
	var n := gw * gh
	terrain = PackedFloat32Array(); terrain.resize(n)
	goal_dist = PackedFloat32Array(); goal_dist.resize(n); goal_dist.fill(1e6)
	density = PackedFloat32Array(); density.resize(n)
	fr = PackedFloat32Array(); fr.resize(n)
	fl = PackedFloat32Array(); fl.resize(n)
	fu = PackedFloat32Array(); fu.resize(n)
	fd = PackedFloat32Array(); fd.resize(n)
	gdir_x = PackedFloat32Array(); gdir_x.resize(n)
	gdir_y = PackedFloat32Array(); gdir_y.resize(n)
	out_vx = PackedFloat32Array(); out_vx.resize(n)
	out_vy = PackedFloat32Array(); out_vy.resize(n)


func idx(x: int, y: int) -> int:
	return y * gw + x


# ── Setup ────────────────────────────────────────────────────────────────

func add_wall(x0: int, y0: int, x1: int, y1: int) -> void:
	for yy in range(maxi(y0, 0), mini(y1, gh)):
		for xx in range(maxi(x0, 0), mini(x1, gw)):
			terrain[idx(xx, yy)] = 1.0


func build_goal_field(goals: Array[Vector2i]) -> void:
	goal_dist.fill(1e6)
	var q: Array[int] = []
	for g in goals:
		if g.x >= 0 and g.x < gw and g.y >= 0 and g.y < gh:
			if terrain[idx(g.x, g.y)] < 0.5:
				goal_dist[idx(g.x, g.y)] = 0.0
				q.append(idx(g.x, g.y))

	var head := 0
	var DX: Array[int] = [1, -1, 0, 0]
	var DY: Array[int] = [0, 0, 1, -1]
	while head < q.size():
		var ci := q[head]; head += 1
		var cx := ci % gw
		var cy := ci / gw
		var cd := goal_dist[ci]
		for d in range(4):
			var nx := cx + DX[d]
			var ny := cy + DY[d]
			if nx < 0 or nx >= gw or ny < 0 or ny >= gh:
				continue
			var ni := idx(nx, ny)
			if terrain[ni] > 0.5:
				continue
			if cd + 1.0 < goal_dist[ni]:
				goal_dist[ni] = cd + 1.0
				q.append(ni)

	# Cap unreachable cells so they don't create extreme gradient artifacts
	var max_d := 0.0
	var n := gw * gh
	for i in range(n):
		if goal_dist[i] < 1e5:
			max_d = maxf(max_d, goal_dist[i])
	for i in range(n):
		if goal_dist[i] >= 1e5:
			goal_dist[i] = max_d + 1.0

	_compute_goal_gradient()


func _compute_goal_gradient() -> void:
	for y in range(gh):
		for x in range(gw):
			var i := idx(x, y)
			if terrain[i] > 0.5:
				gdir_x[i] = 0.0; gdir_y[i] = 0.0
				continue
			var ddx := 0.0
			var ddy := 0.0
			if x > 0 and x < gw - 1:
				ddx = goal_dist[idx(x + 1, y)] - goal_dist[idx(x - 1, y)]
			elif x == 0 and gw > 1:
				ddx = goal_dist[idx(1, y)] - goal_dist[i]
			elif x == gw - 1 and gw > 1:
				ddx = goal_dist[i] - goal_dist[idx(gw - 2, y)]
			if y > 0 and y < gh - 1:
				ddy = goal_dist[idx(x, y + 1)] - goal_dist[idx(x, y - 1)]
			elif y == 0 and gh > 1:
				ddy = goal_dist[idx(x, 1)] - goal_dist[i]
			elif y == gh - 1 and gh > 1:
				ddy = goal_dist[i] - goal_dist[idx(x, gh - 2)]
			var l := sqrt(ddx * ddx + ddy * ddy)
			if l > 1e-6:
				gdir_x[i] = -ddx / l
				gdir_y[i] = -ddy / l
			else:
				gdir_x[i] = 0.0; gdir_y[i] = 0.0


# ── Per-frame density ────────────────────────────────────────────────────

func clear_density() -> void:
	density.fill(0.0)


func splat_agent(world_pos: Vector2) -> void:
	var gx := world_pos.x / cs
	var gy := world_pos.y / cs
	var ix := int(floorf(gx))
	var iy := int(floorf(gy))
	var fx := gx - float(ix)
	var fy := gy - float(iy)
	_add_d(ix,     iy,     (1.0 - fx) * (1.0 - fy))
	_add_d(ix + 1, iy,     fx * (1.0 - fy))
	_add_d(ix,     iy + 1, (1.0 - fx) * fy)
	_add_d(ix + 1, iy + 1, fx * fy)


func _add_d(x: int, y: int, v: float) -> void:
	if x >= 0 and x < gw and y >= 0 and y < gh:
		density[idx(x, y)] += v


# ── SWE flux step ───────────────────────────────────────────────────────

func update_swe(dt: float) -> void:
	var L := cs
	var g := gravity
	var dmp := damping
	var ds := density_scale
	var gs := goal_scale
	var ws := wall_scale
	var mf := max_flux

	for y in range(gh):
		for x in range(gw):
			var i := idx(x, y)
			var H := density[i] * ds + goal_dist[i] * gs + terrain[i] * ws

			if x < gw - 1:
				var j := i + 1
				var Hj := density[j] * ds + goal_dist[j] * gs + terrain[j] * ws
				fr[i] = clampf(dmp * fr[i] + dt * g * (H - Hj) / L, 0.0, mf)
			else:
				fr[i] = 0.0

			if x > 0:
				var j := i - 1
				var Hj := density[j] * ds + goal_dist[j] * gs + terrain[j] * ws
				fl[i] = clampf(dmp * fl[i] + dt * g * (H - Hj) / L, 0.0, mf)
			else:
				fl[i] = 0.0

			if y < gh - 1:
				var j := i + gw
				var Hj := density[j] * ds + goal_dist[j] * gs + terrain[j] * ws
				fd[i] = clampf(dmp * fd[i] + dt * g * (H - Hj) / L, 0.0, mf)
			else:
				fd[i] = 0.0

			if y > 0:
				var j := i - gw
				var Hj := density[j] * ds + goal_dist[j] * gs + terrain[j] * ws
				fu[i] = clampf(dmp * fu[i] + dt * g * (H - Hj) / L, 0.0, mf)
			else:
				fu[i] = 0.0


# ── Velocity computation ────────────────────────────────────────────────

func compute_velocity() -> void:
	var spd := agent_speed
	var ds_slow := density_slowdown

	for y in range(gh):
		for x in range(gw):
			var i := idx(x, y)
			if terrain[i] > 0.5:
				out_vx[i] = 0.0; out_vy[i] = 0.0
				continue

			var fx := fr[i] - fl[i]
			var fy := fd[i] - fu[i]
			var fmag := sqrt(fx * fx + fy * fy)

			var dir_x: float
			var dir_y: float
			if fmag > 0.001:
				dir_x = fx / fmag
				dir_y = fy / fmag
			else:
				dir_x = gdir_x[i]
				dir_y = gdir_y[i]

			var local_speed := spd / (1.0 + density[i] * ds_slow)
			out_vx[i] = dir_x * local_speed
			out_vy[i] = dir_y * local_speed


# ── Sampling ─────────────────────────────────────────────────────────────

func sample_velocity(world_pos: Vector2) -> Vector2:
	var gx := clampf(world_pos.x / cs, 0.5, float(gw) - 1.5)
	var gy := clampf(world_pos.y / cs, 0.5, float(gh) - 1.5)
	var ix := int(floorf(gx))
	var iy := int(floorf(gy))
	ix = clampi(ix, 0, gw - 2)
	iy = clampi(iy, 0, gh - 2)
	var fx := gx - float(ix)
	var fy := gy - float(iy)
	var i00 := idx(ix, iy)
	var i10 := i00 + 1
	var i01 := i00 + gw
	var i11 := i01 + 1
	var w00 := (1.0 - fx) * (1.0 - fy)
	var w10 := fx * (1.0 - fy)
	var w01 := (1.0 - fx) * fy
	var w11 := fx * fy
	return Vector2(
		out_vx[i00] * w00 + out_vx[i10] * w10 + out_vx[i01] * w01 + out_vx[i11] * w11,
		out_vy[i00] * w00 + out_vy[i10] * w10 + out_vy[i01] * w01 + out_vy[i11] * w11
	)
