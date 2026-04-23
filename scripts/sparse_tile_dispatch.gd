class_name SparseTileDispatch
extends RefCounted
## Reusable GPU sparse tile dispatch system for Godot 4.x.
##
## Compacts active tiles into a linear list and generates indirect dispatch args,
## allowing consumer shaders to skip empty regions via thread remapping.
##
## Usage:
##   var std = SparseTileDispatch.new(rd, 800, 480, 8)
##   # Provide the activity & mask buffers once:
##   std.set_activity_buffers(my_density_buf, my_terrain_buf)
##
##   # Each frame:
##   std.reset_counter()                          # before compute_list_begin
##   var cl = rd.compute_list_begin()
##   std.dispatch_compact(cl)                     # produce tile list + indirect args
##   # ... consumer dispatches using dispatch_indirect ...
##   my_shader_dispatch_indirect(cl, std.buf_indirect_args)
##
##   # Readback (optional, for debug / HUD):
##   var n = std.readback_tile_count()
##   var tiles = std.readback_tile_coords(n)

var rd: RenderingDevice

var grid_w: int
var grid_h: int
var tile_size: int
var expand_pixels: int
var threshold: float

var groups_x: int
var groups_y: int
var max_tiles: int

# GPU resources
var buf_compact_counter: RID
var buf_compact_tile_coords: RID
var buf_indirect_args: RID

var _shader_compact: RID
var _pipeline_compact: RID
var _shader_finalize: RID
var _pipeline_finalize: RID
var _uset_compact: RID
var _uset_finalize: RID

var _activity_buf: RID
var _mask_buf: RID


func _init(rendering_device: RenderingDevice, gw: int, gh: int,
		   p_tile_size: int = 8, p_expand: int = 1, p_threshold: float = 0.001) -> void:
	rd = rendering_device
	grid_w = gw
	grid_h = gh
	tile_size = p_tile_size
	expand_pixels = p_expand
	threshold = p_threshold

	groups_x = ceili(float(gw) / float(tile_size))
	groups_y = ceili(float(gh) / float(tile_size))
	max_tiles = groups_x * groups_y

	_create_buffers()
	_load_shaders()


func set_activity_buffers(activity_buf: RID, mask_buf: RID) -> void:
	_activity_buf = activity_buf
	_mask_buf = mask_buf
	_build_uniform_sets()


# ── Per-frame API ────────────────────────────────────────────────────────

func reset_counter() -> void:
	var z4 := PackedByteArray(); z4.resize(4); z4.fill(0)
	rd.buffer_update(buf_compact_counter, 0, 4, z4)


func dispatch_compact(cl: int) -> void:
	var push := PackedByteArray(); push.resize(16)
	push.encode_s32(0, grid_w)
	push.encode_s32(4, grid_h)
	push.encode_s32(8, expand_pixels)
	push.encode_float(12, threshold)

	# Pass 1: scan tiles, emit active coords
	rd.compute_list_bind_compute_pipeline(cl, _pipeline_compact)
	rd.compute_list_bind_uniform_set(cl, _uset_compact, 0)
	rd.compute_list_set_push_constant(cl, push, 16)
	rd.compute_list_dispatch(cl, groups_x, groups_y, 1)
	rd.compute_list_add_barrier(cl)

	# Pass 2: write indirect dispatch args
	var dummy_push := PackedByteArray(); dummy_push.resize(16); dummy_push.fill(0)
	rd.compute_list_bind_compute_pipeline(cl, _pipeline_finalize)
	rd.compute_list_bind_uniform_set(cl, _uset_finalize, 0)
	rd.compute_list_set_push_constant(cl, dummy_push, 16)
	rd.compute_list_dispatch(cl, 1, 1, 1)
	rd.compute_list_add_barrier(cl)


func dispatch_indirect(cl: int) -> void:
	rd.compute_list_dispatch_indirect(cl, buf_indirect_args, 0)


# ── Readback (debug / HUD) ──────────────────────────────────────────────

func readback_tile_count() -> int:
	var data := rd.buffer_get_data(buf_compact_counter, 0, 4)
	return data.decode_u32(0)


func readback_tile_coords(count: int) -> PackedInt32Array:
	if count <= 0:
		return PackedInt32Array()
	var clamped := mini(count, max_tiles)
	var bytes := rd.buffer_get_data(buf_compact_tile_coords, 0, clamped * 4)
	var result := PackedInt32Array()
	result.resize(clamped)
	for i in range(clamped):
		result[i] = bytes.decode_u32(i * 4)
	return result


# ── Internal ─────────────────────────────────────────────────────────────

func _create_buffers() -> void:
	var z4 := PackedByteArray(); z4.resize(4); z4.fill(0)
	buf_compact_counter = rd.storage_buffer_create(4, z4)

	var zt := PackedByteArray(); zt.resize(max_tiles * 4); zt.fill(0)
	buf_compact_tile_coords = rd.storage_buffer_create(max_tiles * 4, zt)

	var z12 := PackedByteArray(); z12.resize(12); z12.fill(0)
	buf_indirect_args = rd.storage_buffer_create(12, z12,
		RenderingDevice.STORAGE_BUFFER_USAGE_DISPATCH_INDIRECT)


func _load_shaders() -> void:
	var compact_spirv := (load("res://shaders/std_compact_generic.glsl") as RDShaderFile).get_spirv()
	var err := compact_spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if err != "":
		push_error("std_compact_generic.glsl COMPILE ERROR: " + err)
	_shader_compact = rd.shader_create_from_spirv(compact_spirv)
	_pipeline_compact = rd.compute_pipeline_create(_shader_compact)

	var fin_spirv := (load("res://shaders/std_finalize_compact.glsl") as RDShaderFile).get_spirv()
	err = fin_spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if err != "":
		push_error("std_finalize_compact.glsl COMPILE ERROR: " + err)
	_shader_finalize = rd.shader_create_from_spirv(fin_spirv)
	_pipeline_finalize = rd.compute_pipeline_create(_shader_finalize)


func _build_uniform_sets() -> void:
	_uset_compact = rd.uniform_set_create([
		_ubuf(0, _activity_buf),
		_ubuf(1, _mask_buf),
		_ubuf(2, buf_compact_counter),
		_ubuf(3, buf_compact_tile_coords),
	], _shader_compact, 0)

	_uset_finalize = rd.uniform_set_create([
		_ubuf(0, buf_compact_counter),
		_ubuf(1, buf_indirect_args),
	], _shader_finalize, 0)


func _ubuf(binding: int, buf: RID) -> RDUniform:
	var u := RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u.binding = binding
	u.add_id(buf)
	return u


func cleanup() -> void:
	if rd == null:
		return
	for rid in [_uset_compact, _uset_finalize,
				_pipeline_compact, _pipeline_finalize,
				_shader_compact, _shader_finalize,
				buf_compact_counter, buf_compact_tile_coords, buf_indirect_args]:
		if rid.is_valid():
			rd.free_rid(rid)
	rd = null
