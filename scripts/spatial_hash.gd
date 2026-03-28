class_name SpatialHash2D
extends RefCounted
## Cell-based spatial hash for O(n) neighbor queries.

var cell_size: float
var cells: Dictionary


func _init(cs: float) -> void:
	cell_size = cs


func clear() -> void:
	cells.clear()


func insert(index: int, pos: Vector2) -> void:
	var key := Vector2i(int(floorf(pos.x / cell_size)), int(floorf(pos.y / cell_size)))
	if cells.has(key):
		cells[key].append(index)
	else:
		cells[key] = [index]


func query(pos: Vector2, radius: float) -> Array:
	var result := []
	var r_cells := int(ceilf(radius / cell_size))
	var cx := int(floorf(pos.x / cell_size))
	var cy := int(floorf(pos.y / cell_size))
	for dy in range(-r_cells, r_cells + 1):
		for dx in range(-r_cells, r_cells + 1):
			var key := Vector2i(cx + dx, cy + dy)
			if cells.has(key):
				result.append_array(cells[key])
	return result
