class_name GPUGI
extends RefCounted

## Radiance Cascades GI driven by an agent emission map.
##
## Per GI update:
## 1. Clear the emission image.
## 2. Stamp walls into the same alpha mask as black blockers.
## 3. Splat every live agent into the emission image.
## 4. Stamp short ranged-attack laser segments into the emission image.
## 5. Build a distance field from the emission/blocker alpha mask with JFA.
## 6. Run the copied 6-face Radiance Cascades pass.
## 7. Resolve cascade 0 into a raw GI image.
## 8. Denoise and temporally stabilize the display image.

const DEFAULT_CASCADE_SIZE := Vector2i(1024, 1024)
const DEFAULT_NUM_CASCADES := 6
const DEFAULT_MERGE_FIX := 4

var rd: RenderingDevice

var tex_w: int
var tex_h: int
var cascade_size := DEFAULT_CASCADE_SIZE
var num_cascades := DEFAULT_NUM_CASCADES
var merge_fix := DEFAULT_MERGE_FIX
var jfa_passes: int

var consts_buf: RID
var emit_params_buf: RID
var image_params_buf: RID
var temporal_params_buf: RID
var jfa_params_bufs: Array[RID] = []
var cube_params_bufs: Array[RID] = []

var tex_emit: RID
var tex_output: RID
var tex_history: RID
var tex_stable: RID
var tex_display: RID
var tex_jfa: RID
var tex_jfa_prev: RID
var tex_distance: RID
var cascade_textures: Array[RID] = []

var shd_clear: RID
var shd_emit: RID
var shd_laser_emit: RID
var shd_seed: RID
var shd_jfa: RID
var shd_distance: RID
var shd_wall_mask: RID
var shd_cube_a: RID
var shd_image: RID
var shd_stabilize: RID
var shd_resample: RID

var pip_clear: RID
var pip_emit: RID
var pip_laser_emit: RID
var pip_seed: RID
var pip_jfa: RID
var pip_distance: RID
var pip_wall_mask: RID
var pip_cube_a: RID
var pip_image: RID
var pip_stabilize: RID
var pip_resample: RID

var buf_pos_x: RID
var buf_pos_y: RID
var buf_agent_info: RID
var buf_fac_colors: RID
var buf_hit_flash: RID
var buf_laser_lines: RID
var buf_laser_ttl: RID
var buf_terrain: RID
var terrain_grid_w: int
var terrain_grid_h: int
var terrain_cell_size: float
var laser_width := 0.65
var laser_emit_strength := 3.2
var laser_life := 0.12
var interval_scale := 1.0
var temporal_blend := 0.82
var spatial_filter_strength := 0.65
var temporal_valid := false
var _last_region_origin := Vector2.ZERO
var _last_region_size := Vector2.ZERO

# Debug: 0 = emission map, 1 = JFA seed/result, 2 = distance field,
# 3 = first cascade face, 4 = final GI.
var debug_stage: int = 4

var _frame_uniform_sets: Array[RID] = []


func _init(rendering_device: RenderingDevice, width: int, height: int,
		p_cascade_size: Vector2i = DEFAULT_CASCADE_SIZE,
		p_num_cascades: int = DEFAULT_NUM_CASCADES,
		p_merge_fix: int = DEFAULT_MERGE_FIX) -> void:
	rd = rendering_device
	tex_w = width
	tex_h = height
	cascade_size = p_cascade_size
	num_cascades = clampi(p_num_cascades, 1, 6)
	merge_fix = clampi(p_merge_fix, 0, 5)
	jfa_passes = ceili(log(float(maxi(tex_w, tex_h))) / log(2.0))

	var cb := PackedInt32Array([0]).to_byte_array()
	cb.resize(16)
	consts_buf = rd.storage_buffer_create(cb.size(), cb)
	emit_params_buf = rd.storage_buffer_create(64, _zero_bytes(16))
	temporal_params_buf = rd.storage_buffer_create(16, _zero_bytes(4))
	_create_static_param_buffers()

	tex_emit = _create_tex(tex_w, tex_h)
	tex_output = _create_tex(tex_w, tex_h)
	tex_history = _create_tex(tex_w, tex_h)
	tex_stable = _create_tex(tex_w, tex_h)
	tex_display = tex_output
	tex_jfa = _create_tex(tex_w, tex_h)
	tex_jfa_prev = _create_tex(tex_w, tex_h)
	tex_distance = _create_tex(tex_w, tex_h)

	cascade_textures.clear()
	for _i in range(6):
		cascade_textures.append(_create_tex(cascade_size.x, cascade_size.y))

	_load_shaders()


func set_agent_buffers(px: RID, py: RID, info: RID, colors: RID, hit_flash: RID) -> void:
	buf_pos_x = px
	buf_pos_y = py
	buf_agent_info = info
	buf_fac_colors = colors
	buf_hit_flash = hit_flash


func set_laser_buffers(lines: RID, ttl: RID) -> void:
	buf_laser_lines = lines
	buf_laser_ttl = ttl


func set_terrain_buffer(terrain: RID, grid_w: int, grid_h: int, cell_size: float) -> void:
	buf_terrain = terrain
	terrain_grid_w = grid_w
	terrain_grid_h = grid_h
	terrain_cell_size = cell_size


func compute_gi(agent_count: int, region_origin: Vector2, region_size: Vector2,
		splat_radius: float, emit_strength: float) -> void:
	if not _has_agent_buffers():
		return

	var region_stable := temporal_valid and _is_same_region(region_origin, region_size)
	_update_temporal_params(temporal_blend if region_stable else 0.0)
	_update_emit_params(agent_count, region_origin, region_size, splat_radius, emit_strength)
	_update_interval_params()
	_last_region_origin = region_origin
	_last_region_size = region_size

	_frame_uniform_sets.clear()
	var cl := rd.compute_list_begin()

	_dispatch_clear(cl, tex_emit)
	rd.compute_list_add_barrier(cl)

	_dispatch_wall_mask(cl)
	rd.compute_list_add_barrier(cl)

	_dispatch_emit(cl, agent_count, region_origin, region_size, splat_radius, emit_strength)
	rd.compute_list_add_barrier(cl)

	_dispatch_laser_emit(cl, agent_count)
	rd.compute_list_add_barrier(cl)

	if debug_stage == 0:
		_dispatch_resample(cl, tex_emit)
		tex_display = tex_output
		_finish_compute(cl)
		return

	_dispatch_seed(cl)
	rd.compute_list_add_barrier(cl)

	if debug_stage == 1:
		_dispatch_resample(cl, tex_jfa)
		tex_display = tex_output
		_finish_compute(cl)
		return

	_dispatch_jfa(cl)
	rd.compute_list_add_barrier(cl)

	_dispatch_distance(cl)
	rd.compute_list_add_barrier(cl)

	if debug_stage == 2:
		_dispatch_resample(cl, tex_distance)
		tex_display = tex_output
		_finish_compute(cl)
		return

	_dispatch_radiance_cascades(cl)

	if debug_stage == 3:
		_dispatch_resample(cl, cascade_textures[0])
		tex_display = tex_output
	else:
		_dispatch_image(cl)
		rd.compute_list_add_barrier(cl)
		_dispatch_stabilize(cl)

	_finish_compute(cl)
	if debug_stage == 4:
		_swap_temporal_output()
		temporal_valid = true


func get_output_image() -> Image:
	var source := tex_display if tex_display.is_valid() else tex_output
	var raw := rd.texture_get_data(source, 0)
	return Image.create_from_data(tex_w, tex_h, false, Image.FORMAT_RGBAF, raw)


func cleanup() -> void:
	if rd == null:
		return

	var free_rid := func(rid: RID) -> void:
		if rid.is_valid():
			rd.free_rid(rid)

	for us in _frame_uniform_sets:
		free_rid.call(us)
	_frame_uniform_sets.clear()

	for rid in [
		tex_emit, tex_output, tex_history, tex_stable, tex_jfa, tex_jfa_prev, tex_distance,
		consts_buf, emit_params_buf, image_params_buf, temporal_params_buf,
		pip_clear, pip_emit, pip_laser_emit, pip_seed, pip_jfa, pip_distance, pip_wall_mask, pip_cube_a, pip_image, pip_stabilize, pip_resample,
		shd_clear, shd_emit, shd_laser_emit, shd_seed, shd_jfa, shd_distance, shd_wall_mask, shd_cube_a, shd_image, shd_stabilize, shd_resample,
	]:
		free_rid.call(rid)

	for rid in cascade_textures:
		free_rid.call(rid)
	for rid in jfa_params_bufs:
		free_rid.call(rid)
	for rid in cube_params_bufs:
		free_rid.call(rid)

	rd = null


func _has_agent_buffers() -> bool:
	return buf_pos_x.is_valid() and buf_pos_y.is_valid() and buf_agent_info.is_valid() and buf_fac_colors.is_valid() and buf_hit_flash.is_valid()


func _has_laser_buffers() -> bool:
	return buf_laser_lines.is_valid() and buf_laser_ttl.is_valid()


func _dispatch_clear(cl: int, tex: RID) -> void:
	var us := _uniform_set([_img_uniform(0, tex)], shd_clear)
	rd.compute_list_bind_compute_pipeline(cl, pip_clear)
	rd.compute_list_bind_uniform_set(cl, us, 0)
	rd.compute_list_dispatch(cl, ceili(float(_texture_width(tex)) / 16.0), ceili(float(_texture_height(tex)) / 16.0), 1)


func _dispatch_emit(cl: int, agent_count: int, region_origin: Vector2, region_size: Vector2,
		splat_radius: float, emit_strength: float) -> void:
	var us := _uniform_set([
		_buf_uniform(0, buf_pos_x),
		_buf_uniform(1, buf_pos_y),
		_buf_uniform(2, buf_agent_info),
		_buf_uniform(3, buf_fac_colors),
		_img_uniform(4, tex_emit),
		_buf_uniform(5, emit_params_buf),
		_buf_uniform(6, buf_hit_flash),
	], shd_emit)

	rd.compute_list_bind_compute_pipeline(cl, pip_emit)
	rd.compute_list_bind_uniform_set(cl, us, 0)
	rd.compute_list_dispatch(cl, ceili(float(agent_count) / 64.0), 1, 1)


func _dispatch_laser_emit(cl: int, agent_count: int) -> void:
	if not _has_laser_buffers():
		return
	var us := _uniform_set([
		_buf_uniform(0, buf_agent_info),
		_buf_uniform(1, buf_fac_colors),
		_buf_uniform(2, buf_laser_lines),
		_buf_uniform(3, buf_laser_ttl),
		_img_uniform(4, tex_emit),
		_buf_uniform(5, emit_params_buf),
	], shd_laser_emit)

	rd.compute_list_bind_compute_pipeline(cl, pip_laser_emit)
	rd.compute_list_bind_uniform_set(cl, us, 0)
	rd.compute_list_dispatch(cl, ceili(float(agent_count) / 64.0), 1, 1)


func _dispatch_wall_mask(cl: int) -> void:
	if not buf_terrain.is_valid():
		return
	var us := _uniform_set([
		_buf_uniform(0, buf_terrain),
		_img_uniform(1, tex_emit),
		_buf_uniform(2, emit_params_buf),
	], shd_wall_mask)

	rd.compute_list_bind_compute_pipeline(cl, pip_wall_mask)
	rd.compute_list_bind_uniform_set(cl, us, 0)
	rd.compute_list_dispatch(cl, ceili(float(tex_w) / 16.0), ceili(float(tex_h) / 16.0), 1)


func _dispatch_seed(cl: int) -> void:
	var us := _uniform_set([
		_buf_uniform(0, consts_buf),
		_img_uniform(1, tex_jfa),
		_img_uniform(2, tex_emit),
	], shd_seed)

	rd.compute_list_bind_compute_pipeline(cl, pip_seed)
	rd.compute_list_bind_uniform_set(cl, us, 0)
	rd.compute_list_dispatch(cl, ceili(float(tex_w) / 16.0), ceili(float(tex_h) / 16.0), 1)


func _dispatch_jfa(cl: int) -> void:
	for i in range(jfa_passes - 1, -1, -1):
		_swap_jfa()
		var us := _uniform_set([
			_buf_uniform(0, consts_buf),
			_img_uniform(1, tex_jfa),
			_img_uniform(2, tex_jfa_prev),
			_buf_uniform(4, jfa_params_bufs[i]),
		], shd_jfa)

		rd.compute_list_bind_compute_pipeline(cl, pip_jfa)
		rd.compute_list_bind_uniform_set(cl, us, 0)
		rd.compute_list_dispatch(cl, ceili(float(tex_w) / 16.0), ceili(float(tex_h) / 16.0), 1)
		rd.compute_list_add_barrier(cl)


func _dispatch_distance(cl: int) -> void:
	var us := _uniform_set([
		_buf_uniform(0, consts_buf),
		_img_uniform(1, tex_distance),
		_img_uniform(2, tex_jfa),
	], shd_distance)

	rd.compute_list_bind_compute_pipeline(cl, pip_distance)
	rd.compute_list_bind_uniform_set(cl, us, 0)
	rd.compute_list_dispatch(cl, ceili(float(tex_w) / 16.0), ceili(float(tex_h) / 16.0), 1)


func _dispatch_radiance_cascades(cl: int) -> void:
	for face in range(num_cascades - 1, -1, -1):
		var us := _uniform_set(_cascade_uniforms([
			_buf_uniform(0, consts_buf),
			_buf_uniform(1, cube_params_bufs[face]),
			_img_uniform(2, tex_emit),
			_img_uniform(3, tex_distance),
		]), shd_cube_a)

		rd.compute_list_bind_compute_pipeline(cl, pip_cube_a)
		rd.compute_list_bind_uniform_set(cl, us, 0)
		rd.compute_list_dispatch(cl, ceili(float(cascade_size.x) / 16.0), ceili(float(cascade_size.y) / 16.0), 1)
		rd.compute_list_add_barrier(cl)


func _dispatch_image(cl: int) -> void:
	var us := _uniform_set(_cascade_uniforms([
		_buf_uniform(0, consts_buf),
		_img_uniform(1, tex_output),
		_img_uniform(2, tex_emit),
		_buf_uniform(3, image_params_buf),
	]), shd_image)

	rd.compute_list_bind_compute_pipeline(cl, pip_image)
	rd.compute_list_bind_uniform_set(cl, us, 0)
	rd.compute_list_dispatch(cl, ceili(float(tex_w) / 16.0), ceili(float(tex_h) / 16.0), 1)


func _dispatch_stabilize(cl: int) -> void:
	var us := _uniform_set([
		_img_uniform(0, tex_output),
		_img_uniform(1, tex_history),
		_img_uniform(2, tex_stable),
		_buf_uniform(3, temporal_params_buf),
	], shd_stabilize)

	rd.compute_list_bind_compute_pipeline(cl, pip_stabilize)
	rd.compute_list_bind_uniform_set(cl, us, 0)
	rd.compute_list_dispatch(cl, ceili(float(tex_w) / 16.0), ceili(float(tex_h) / 16.0), 1)


func _dispatch_resample(cl: int, source_tex: RID) -> void:
	var us := _uniform_set([
		_img_uniform(1, tex_output),
		_img_uniform(2, source_tex),
	], shd_resample)

	rd.compute_list_bind_compute_pipeline(cl, pip_resample)
	rd.compute_list_bind_uniform_set(cl, us, 0)
	rd.compute_list_dispatch(cl, ceili(float(tex_w) / 16.0), ceili(float(tex_h) / 16.0), 1)


func _finish_compute(cl: int) -> void:
	rd.compute_list_end()
	rd.submit()
	rd.sync()

	for us in _frame_uniform_sets:
		if us.is_valid():
			rd.free_rid(us)
	_frame_uniform_sets.clear()


func _swap_jfa() -> void:
	var tmp := tex_jfa
	tex_jfa = tex_jfa_prev
	tex_jfa_prev = tmp


func _swap_temporal_output() -> void:
	var tmp := tex_history
	tex_history = tex_stable
	tex_stable = tmp
	tex_display = tex_history


func _is_same_region(region_origin: Vector2, region_size: Vector2) -> bool:
	if _last_region_size == Vector2.ZERO:
		return false
	return (
		_last_region_origin.distance_to(region_origin) < 0.25
		and _last_region_size.distance_to(region_size) < 0.25
	)


func _create_static_param_buffers() -> void:
	jfa_params_bufs.clear()
	for i in range(jfa_passes):
		var jfa_bytes := PackedFloat32Array([pow(2.0, i)]).to_byte_array()
		jfa_bytes.resize(16)
		jfa_params_bufs.append(rd.storage_buffer_create(jfa_bytes.size(), jfa_bytes))

	cube_params_bufs.clear()
	for face in range(6):
		var cube_bytes := PackedByteArray()
		cube_bytes.resize(32)
		cube_bytes.encode_s32(0, face)
		cube_bytes.encode_s32(4, num_cascades)
		cube_bytes.encode_s32(8, cascade_size.x)
		cube_bytes.encode_s32(12, cascade_size.y)
		cube_bytes.encode_s32(16, merge_fix)
		cube_bytes.encode_float(20, interval_scale)
		cube_params_bufs.append(rd.storage_buffer_create(cube_bytes.size(), cube_bytes))

	var image_bytes := PackedByteArray()
	image_bytes.resize(16)
	image_bytes.encode_s32(0, num_cascades)
	image_bytes.encode_s32(4, cascade_size.x)
	image_bytes.encode_s32(8, cascade_size.y)
	image_bytes.encode_float(12, interval_scale)
	image_params_buf = rd.storage_buffer_create(image_bytes.size(), image_bytes)


func _update_emit_params(agent_count: int, region_origin: Vector2, region_size: Vector2,
		splat_radius: float, emit_strength: float) -> void:
	var bytes := PackedByteArray()
	bytes.resize(64)
	bytes.encode_s32(0, agent_count)
	bytes.encode_float(4, float(tex_w))
	bytes.encode_float(8, float(tex_h))
	bytes.encode_float(12, region_origin.x)
	bytes.encode_float(16, region_origin.y)
	bytes.encode_float(20, region_size.x)
	bytes.encode_float(24, region_size.y)
	bytes.encode_float(28, splat_radius)
	bytes.encode_float(32, emit_strength)
	bytes.encode_s32(36, terrain_grid_w)
	bytes.encode_s32(40, terrain_grid_h)
	bytes.encode_float(44, terrain_cell_size)
	bytes.encode_float(48, 1.0)
	bytes.encode_float(52, laser_width)
	bytes.encode_float(56, laser_emit_strength)
	bytes.encode_float(60, laser_life)
	rd.buffer_update(emit_params_buf, 0, bytes.size(), bytes)


func _update_temporal_params(history_weight: float) -> void:
	var bytes := PackedByteArray()
	bytes.resize(16)
	bytes.encode_float(0, clampf(history_weight, 0.0, 0.98))
	bytes.encode_float(4, clampf(spatial_filter_strength, 0.0, 1.0))
	bytes.encode_s32(8, 0)
	bytes.encode_s32(12, 0)
	rd.buffer_update(temporal_params_buf, 0, bytes.size(), bytes)


func _update_interval_params() -> void:
	var bytes := PackedByteArray()
	bytes.resize(4)
	bytes.encode_float(0, clampf(interval_scale, 0.125, 8.0))
	if image_params_buf.is_valid():
		rd.buffer_update(image_params_buf, 12, 4, bytes)
	for buf in cube_params_bufs:
		if buf.is_valid():
			rd.buffer_update(buf, 20, 4, bytes)


func _cascade_uniforms(base_uniforms: Array) -> Array:
	var uniforms := base_uniforms.duplicate()
	for i in range(6):
		uniforms.append(_img_uniform(10 + i, cascade_textures[i]))
	return uniforms


func _uniform_set(uniforms: Array, shader: RID) -> RID:
	var us := rd.uniform_set_create(uniforms, shader, 0)
	_frame_uniform_sets.append(us)
	return us


func _create_tex(w: int, h: int) -> RID:
	var fmt := RDTextureFormat.new()
	fmt.width = w
	fmt.height = h
	fmt.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	fmt.usage_bits = (
		RenderingDevice.TEXTURE_USAGE_STORAGE_BIT
		| RenderingDevice.TEXTURE_USAGE_CAN_COPY_TO_BIT
		| RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT
	)
	return rd.texture_create(fmt, RDTextureView.new())


func _zero_bytes(count: int) -> PackedByteArray:
	var bytes := PackedByteArray()
	bytes.resize(count * 4)
	bytes.fill(0)
	return bytes


func _load_shaders() -> void:
	shd_clear = _load_shader("res://shaders/gi_clear.glsl")
	shd_emit = _load_shader("res://shaders/gi_agent_emit.glsl")
	shd_laser_emit = _load_shader("res://shaders/gi_laser_emit.glsl")
	shd_seed = _load_shader("res://shaders/gi_seed.glsl")
	shd_jfa = _load_shader("res://shaders/gi_jump_flood_algorithm.glsl")
	shd_distance = _load_shader("res://shaders/gi_distance.glsl")
	shd_wall_mask = _load_shader("res://shaders/gi_wall_mask.glsl")
	shd_cube_a = _load_shader("res://shaders/gi_rc_cube_a.glsl")
	shd_image = _load_shader("res://shaders/gi_rc_image.glsl")
	shd_stabilize = _load_shader("res://shaders/gi_stabilize.glsl")
	shd_resample = _load_shader("res://shaders/gi_resample_image.glsl")

	pip_clear = rd.compute_pipeline_create(shd_clear)
	pip_emit = rd.compute_pipeline_create(shd_emit)
	pip_laser_emit = rd.compute_pipeline_create(shd_laser_emit)
	pip_seed = rd.compute_pipeline_create(shd_seed)
	pip_jfa = rd.compute_pipeline_create(shd_jfa)
	pip_distance = rd.compute_pipeline_create(shd_distance)
	pip_wall_mask = rd.compute_pipeline_create(shd_wall_mask)
	pip_cube_a = rd.compute_pipeline_create(shd_cube_a)
	pip_image = rd.compute_pipeline_create(shd_image)
	pip_stabilize = rd.compute_pipeline_create(shd_stabilize)
	pip_resample = rd.compute_pipeline_create(shd_resample)


func _load_shader(path: String) -> RID:
	var file := load(path) as RDShaderFile
	if file == null:
		push_error("[GPUGI] Could not load shader: " + path)
		return RID()

	var spirv := file.get_spirv()
	var err := spirv.get_stage_compile_error(RenderingDevice.SHADER_STAGE_COMPUTE)
	if not err.is_empty():
		push_error("[GPUGI] Shader compile error in %s:\n%s" % [path, err])
	return rd.shader_create_from_spirv(spirv)


func _img_uniform(binding: int, tex: RID) -> RDUniform:
	var u := RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	u.binding = binding
	u.add_id(tex)
	return u


func _buf_uniform(binding: int, buf: RID) -> RDUniform:
	var u := RDUniform.new()
	u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	u.binding = binding
	u.add_id(buf)
	return u


func _texture_width(tex: RID) -> int:
	if tex == tex_emit or tex == tex_output or tex == tex_history or tex == tex_stable or tex == tex_jfa or tex == tex_jfa_prev or tex == tex_distance:
		return tex_w
	return cascade_size.x


func _texture_height(tex: RID) -> int:
	if tex == tex_emit or tex == tex_output or tex == tex_history or tex == tex_stable or tex == tex_jfa or tex == tex_jfa_prev or tex == tex_distance:
		return tex_h
	return cascade_size.y
