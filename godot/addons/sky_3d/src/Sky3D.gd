# Copyright (c) 2023-2025 Cory Petkovsek and Contributors
# Copyright (c) 2021 J. Cuellar

## Sky3D is an Atmosphereic Day/Night Cycle for Godot 4.
##
## It manages time, moving the sun, moon, and stars, and consolidates environmental lighting controls.
## To use it, remove any WorldEnvironment node from you scene, then add a new Sky3D node.
## Explore and configure the settings in the Sky3D, SunLight, MoonLight, TimeOfDay, and Skydome nodes.

@tool
class_name Sky3D
extends WorldEnvironment

## Emitted when the environment variable has changed.
signal environment_changed

## The Sun DirectionalLight.
var sun: DirectionalLight3D
## The Moon DirectionalLight.
var moon: DirectionalLight3D
## The TimeOfDay node.
var tod: TimeOfDay
## The Skydome node.
var sky: Skydome


## Enables all rendering and time tracking.
@export var sky3d_enabled: bool = true : set = set_sky3d_enabled

func set_sky3d_enabled(value: bool) -> void:
	sky3d_enabled = value
	if value:
		show_sky()
		resume()
	else:
		hide_sky()
		pause()


#####################
## Visibility
#####################

@export_group("Visibility")


## Enables the sky shader. Disable sky, lights, fog for a black sky or call hide_sky().
@export var sky_enabled: bool = true : set = set_sky_enabled

func set_sky_enabled(value: bool) -> void:
	sky_enabled = value
	if not sky:
		return
	sky.sky_visible = value
	sky.clouds_cumulus_visible = clouds_enabled and value


## Enables the Sun and Moon DirectionalLights.
@export var lights_enabled: bool = true : set = set_lights_enabled

func set_lights_enabled(value: bool) -> void:
	lights_enabled = value
	if not sky:
		return
	sky.sun_light_enable = value
	sky.moon_light_enable = value
	sky.__sun_light_node.visible = value && sky.__sun_light_node.light_energy > 0
	sky.__moon_light_node.visible = value && sky.__moon_light_node.light_energy > 0


## Enables the screen space fog shader.
@export var fog_enabled: bool = true : set = set_fog_enabled

func set_fog_enabled(value: bool) -> void:
	fog_enabled = value
	if sky:
		sky.fog_visible = value


## Enables the 2D and cumulus cloud layers.
@export var clouds_enabled: bool = true : set = set_clouds_enabled

func set_clouds_enabled(value: bool) -> void:
	clouds_enabled = value
	if not sky:
		return
	sky.clouds_cumulus_visible = value
	sky.clouds_thickness = float(value) * 1.7
	# TODO should create an on/off in skydome so disabling this doesn't change the enabled value


## Disables rendering of sky, fog, and lights
func hide_sky() -> void:
	sky_enabled = false
	lights_enabled = false
	fog_enabled = false


## Enables rendering of sky, fog, and lights
func show_sky() -> void:
	sky_enabled = true
	lights_enabled = true
	fog_enabled = true


#####################
## Time
#####################

@export_group("Time")


## Move time forward in the editor.
@export var enable_editor_time: bool = true : set = set_editor_time_enabled

func set_editor_time_enabled(value: bool) -> void:
	enable_editor_time = value
	if tod:
		tod.update_in_editor = value


## Move time forward in game.
@export var enable_game_time: bool = true : set = set_game_time_enabled

func set_game_time_enabled(value: bool) -> void:
	enable_game_time = value
	if tod:
		tod.update_in_game = value


## The time right now in hours, 0-23.99. Larger and smaller values will wrap
@export_range(0.0, 23.99, .01, "or_greater", "or_less") var current_time: float = 8.0 : set = set_current_time

func set_current_time(value: float) -> void:
	current_time = value
	if tod and tod.total_hours != current_time:
		tod.total_hours = value


## The length of a full day in real minutes. +/-1440 (24 hours), forward or backwards.
@export_range(-1440,1440,.1) var minutes_per_day: float = 15.0 : set = set_minutes_per_day

func set_minutes_per_day(value):
	minutes_per_day = value
	if tod:
		tod.total_cycle_in_minutes = value


## Frequency of updates. Set to 0.016 for 60fps.
@export_range(0.016, 10.0) var update_interval: float = 0.1 : set = set_update_interval

func set_update_interval(value: float) -> void:
	update_interval = value
	if tod:
		tod.update_interval = value


## Tracks if the sun is above the horizon.
var _is_day: bool = true


## Returns true if the sun is above the horizon.
func is_day() -> bool:
	return _is_day

	
## Returns true if the sun is below the horizon.
func is_night() -> bool:
	return not _is_day


## Pauses time calculation.
func pause() -> void:
	if tod:
		tod.pause()


## Resumes time calculation.
func resume() -> void:
	if tod:
		tod.resume()


func _on_timeofday_updated(time: float) -> void:
	if tod:
		minutes_per_day = tod.total_cycle_in_minutes
		current_time = tod.total_hours
		update_interval = tod.update_interval
	update_day_night()


## Recalculates if it's currently day or night. Adjusts night ambient light if changing state or forced.
func update_day_night(force: bool = false) -> void:
	if not (sky and environment):
		return

	# If day transitioning to night		
	if abs(sky.sun_altitude) > 87 and (_is_day or force):
		_is_day = false
		var tween: Tween = get_tree().create_tween()
		tween.set_parallel(true)
		var contrib: float = minf(night_ambient_min, sky_contribution) if night_ambient else sky_contribution
		tween.tween_property(environment, "ambient_light_sky_contribution", contrib, ambient_tween_time)
		tween.tween_property(environment.sky.sky_material, "energy_multiplier", 1., ambient_tween_time)

	# Else if night transitioning to day		
	elif abs(sky.sun_altitude) <= 87 and (not _is_day or force):
		_is_day = true
		var tween: Tween = get_tree().create_tween()
		tween.set_parallel(true)
		tween.tween_property(environment, "ambient_light_sky_contribution", sky_contribution, ambient_tween_time)
		tween.tween_property(environment.sky.sky_material, "energy_multiplier", reflected_energy, ambient_tween_time)


#####################
## Lighting
#####################

@export_group("Lighting")


## Exposure used for the tonemapper. See Evironment.tonemap_exposure
@export_range(0,16,.005) var tonemap_exposure: float = 1.0: set = set_tonemap_exposure

func set_tonemap_exposure(value: float) -> void:
	if environment:
		tonemap_exposure = value
		environment.tonemap_exposure = value


## Strength of skydome and fog.
@export_range(0,16,.005) var skydome_energy: float = 1.3: set = set_skydome_energy

func set_skydome_energy(value: float) -> void:
	if sky:
		skydome_energy = value
		sky.exposure = value
		sky.clouds_cumulus_intensity = value * .769 # (1/1.3 default sun energy)


## Exposure of camera connected to Environment.camera_attributes.
@export_range(0,16,.005) var camera_exposure: float = 1.0: set = set_camera_exposure

func set_camera_exposure(value: float) -> void:
	if camera_attributes:
		camera_exposure = value
		camera_attributes.exposure_multiplier = value


## Maximum strength of Sun DirectionalLight, visible during the day.
@export_range(0,16,.005) var sun_energy: float = 1.0: set = set_sun_energy
		
func set_sun_energy(value: float) -> void:
	sun_energy = value
	if sky:
		sky.sun_light_energy = value


## Opacity of Sun DirectionalLight shadow.
@export_range(0,1,.005) var sun_shadow_opacity: float = 1.0: set = set_sun_shadow_opacity

func set_sun_shadow_opacity(value: float) -> void:
	sun_shadow_opacity = value
	if sun:
		sun.shadow_opacity = value
		

## Strength of refelcted light from the PhysicalSky. See PhysicalSkyMaterial.energy_multiplier
@export_range(0,128,.005) var reflected_energy: float = 1.0: set = set_reflected_energy

func set_reflected_energy(value: float) -> void:
	if environment:
		reflected_energy = value
		if environment.sky:
			environment.sky.sky_material.energy_multiplier = value

			
## Ratio of ambient light to sky light. See Environment.ambient_light_sky_contribution.
@export_range(0,1,.005) var sky_contribution: float = 1.0: set = set_sky_contribution

func set_sky_contribution(value: float) -> void:
	if environment:
		sky_contribution = value
		environment.ambient_light_sky_contribution = value
		update_day_night(true)


## Strength of ambient light. Works outside of Reflection Probe / GI volumes and sky_contribution < 1.
## See Environment.ambient_light_energy.
@export_range(0,16,.005) var ambient_energy: float = 1.0: set = set_ambient_energy

func set_ambient_energy(value: float) -> void:
	if environment:
		ambient_energy = value
		environment.ambient_light_energy = value
		update_day_night(true)


@export_subgroup("Auto Exposure")


## Enables CameraAttributes.auto_exposure_enabled.
@export var auto_exposure: bool = false: set = set_auto_exposure_enabled

func set_auto_exposure_enabled(value: bool) -> void:
	if camera_attributes:
		auto_exposure = value
		camera_attributes.auto_exposure_enabled = value


## Sets CameraAttributes.auto_exposure_scale.
@export_range(0.01,16,.005) var auto_exposure_scale: float = 0.2: set = set_auto_exposure_scale

func set_auto_exposure_scale(value: float) -> void:
	if camera_attributes:
		auto_exposure_scale = value
		camera_attributes.auto_exposure_scale = value


## Sets CameraAttributesPractical.auto_exposure_min_sensitivity.
@export_range(0,1600,.5) var auto_exposure_min: float = 0.0: set = set_auto_exposure_min

func set_auto_exposure_min(value: float) -> void:
	if camera_attributes:
		auto_exposure_min = value
		camera_attributes.auto_exposure_min_sensitivity = value


## Sets CameraAttributesPractical.auto_exposure_max_sensitivity.
@export_range(30,64000,.5) var auto_exposure_max: float = 800.0: set = set_auto_exposure_max

func set_auto_exposure_max(value: float) -> void:
	if camera_attributes:
		auto_exposure_max = value
		camera_attributes.auto_exposure_max_sensitivity = value


## Sets CameraAttributes.auto_exposure_speed.
@export_range(0.1,64,.1) var auto_exposure_speed: float = 0.5: set = set_auto_exposure_speed

func set_auto_exposure_speed(value: float) -> void:
	if camera_attributes:
		auto_exposure_speed = value
		camera_attributes.auto_exposure_speed = value


@export_subgroup("Night")


## Maximum strength of Moon DirectionalLight, visible at night.
@export_range(0,16,.005) var moon_energy: float = .3: set = set_moon_energy

func set_moon_energy(value: float) -> void:
	moon_energy = value
	if moon:
		sky.moon_light_energy = value


## Opacity of Moon DirectionalLight shadow.
@export_range(0,1,.005) var moon_shadow_opacity: float = 1.0: set = set_moon_shadow_opacity

func set_moon_shadow_opacity(value: float) -> void:
	moon_shadow_opacity = value
	if moon:
		moon.shadow_opacity = value


## Enables a different ambient light setting at night.
@export var night_ambient: bool = true: set = set_night_ambient

func set_night_ambient(value: bool) -> void:
	night_ambient = value
	update_day_night(true)


## Strength of ambient light at night. Sky_contribution must be < 1. See Environment.ambient_light_energy.
@export_range(0,1,.005) var night_ambient_min: float = .7: set = set_night_ambient_min

func set_night_ambient_min(value: float) -> void:
	night_ambient_min = value
	if night_ambient:
		update_day_night(true)


## Transition time for ambient light change, typically transitioning between day and night.
@export_range(0,30,.05) var ambient_tween_time: float = 3.: set = set_ambient_tween_time

func set_ambient_tween_time(value: float) -> void:
	ambient_tween_time = value


#####################
## Setup
#####################


func _notification(what: int) -> void:
	# Must be after _init and before _enter_tree to properly set vars like 'sky' for setters
	if what in [ NOTIFICATION_SCENE_INSTANTIATED, NOTIFICATION_ENTER_TREE ]:
		_initialize()


func _initialize() -> void:
	# Create default environment
	if environment == null:
		environment = Environment.new()
		environment.background_mode = Environment.BG_SKY
		environment.sky = Sky.new()
		environment.sky.sky_material = PhysicalSkyMaterial.new()
		environment.sky.sky_material.use_debanding = false
		environment.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
		environment.ambient_light_sky_contribution = 0.7
		environment.ambient_light_energy = 1.0
		environment.reflected_light_source = Environment.REFLECTION_SOURCE_SKY
		environment.tonemap_mode = Environment.TONE_MAPPER_ACES
		environment.tonemap_white = 6
		emit_signal("environment_changed", environment)
	
	# Create default camera attributes
	if camera_attributes == null:
		camera_attributes = CameraAttributesPractical.new()

	# Assign children nodes
	
	if has_node("SunLight"):
		sun = $SunLight
	elif is_inside_tree():
		sun = DirectionalLight3D.new()
		sun.name = "SunLight"
		add_child(sun, true)
		sun.owner = get_tree().edited_scene_root
		sun.shadow_enabled = true
	
	if has_node("MoonLight"):
		moon = $MoonLight
	elif is_inside_tree():
		moon = DirectionalLight3D.new()
		moon.name = "MoonLight"
		add_child(moon, true)
		moon.owner = get_tree().edited_scene_root
		moon.shadow_enabled = true

	if has_node("Skydome"):
		sky = $Skydome
		sky.environment = environment
	elif is_inside_tree():
		sky = Skydome.new()
		sky.name = "Skydome"
		add_child(sky, true)
		sky.owner = get_tree().edited_scene_root
		sky.sun_light_path = "../SunLight"
		sky.moon_light_path = "../MoonLight"
		sky.environment = environment

	if has_node("TimeOfDay"):
		tod = $TimeOfDay
	elif is_inside_tree():
		tod = TimeOfDay.new()
		tod.name = "TimeOfDay"
		add_child(tod, true)
		tod.owner = get_tree().edited_scene_root
		tod.dome_path = "../Skydome"
	if tod and not tod.time_changed.is_connected(_on_timeofday_updated):
		tod.time_changed.connect(_on_timeofday_updated)


func _enter_tree() -> void:
	update_day_night(true)


func _set(property: StringName, value: Variant) -> bool:
	match property:
		"environment":
			sky.environment = value
			environment = value
			emit_signal("environment_changed", environment)
			return true
	return false


#####################
## Constants
#####################

# Node names
const SKY_INSTANCE:= "_SkyMeshI"
const FOG_INSTANCE:= "_FogMeshI"
const MOON_INSTANCE:= "MoonRender"
const CLOUDS_C_INSTANCE:= "_CloudsCumulusI"

# Shaders
const _sky_shader: Shader = preload("res://addons/sky_3d/shaders/Sky.gdshader")
const _pv_sky_shader: Shader = preload("res://addons/sky_3d/shaders/PerVertexSky.gdshader")
const _clouds_cumulus_shader: Shader = preload("res://addons/sky_3d/shaders/CloudsCumulus.gdshader")
const _fog_shader: Shader = preload("res://addons/sky_3d/shaders/AtmFog.gdshader")

# Scenes
const _moon_render: PackedScene = preload("res://addons/sky_3d/assets/resources/MoonRender.tscn")

# Textures
const _moon_texture: Texture2D = preload("res://addons/sky_3d/assets/thirdparty/textures/moon/MoonMap.png")
const _background_texture: Texture2D = preload("res://addons/sky_3d/assets/thirdparty/textures/milkyway/Milkyway.jpg")
const _stars_field_texture: Texture2D = preload("res://addons/sky_3d/assets/thirdparty/textures/milkyway/StarField.jpg")
const _sun_moon_curve_fade: Curve = preload("res://addons/sky_3d/assets/resources/SunMoonLightFade.tres")
const _stars_field_noise: Texture2D = preload("res://addons/sky_3d/assets/textures/noise.jpg")
const _clouds_texture: Texture2D = preload("res://addons/sky_3d/assets/resources/SNoise.tres")
const _clouds_cumulus_texture: Texture2D = preload("res://addons/sky_3d/assets/textures/noiseClouds.png")

# Skydome
const DEFAULT_POSITION:= Vector3(0.0000001, 0.0000001, 0.0000001)

# Coords
const SUN_DIR_P:= "_sun_direction"
const MOON_DIR_P:= "_moon_direction"
const MOON_MATRIX:= "_moon_matrix"

# General
const TEXTURE_P:= "_texture"
const COLOR_CORRECTION_P:= "_color_correction_params"
const GROUND_COLOR_P:= "_ground_color"
const NOISE_TEX:= "_noise_tex"
const HORIZON_LEVEL = "_horizon_level"

# Atmosphere
const ATM_DARKNESS_P:= "_atm_darkness"
const ATM_BETA_RAY_P:= "_atm_beta_ray"
const ATM_SUN_INTENSITY_P:= "_atm_sun_intensity"
const ATM_DAY_TINT_P:= "_atm_day_tint"
const ATM_HORIZON_LIGHT_TINT_P:= "_atm_horizon_light_tint"

const ATM_NIGHT_TINT_P:= "_atm_night_tint"
const ATM_LEVEL_PARAMS_P:= "_atm_level_params"
const ATM_THICKNESS_P:= "_atm_thickness"
const ATM_BETA_MIE_P:= "_atm_beta_mie"

const ATM_SUN_MIE_TINT_P:= "_atm_sun_mie_tint"
const ATM_SUN_MIE_INTENSITY_P:= "_atm_sun_mie_intensity"
const ATM_SUN_PARTIAL_MIE_PHASE_P:= "_atm_sun_partial_mie_phase"

const ATM_MOON_MIE_TINT_P:= "_atm_moon_mie_tint"
const ATM_MOON_MIE_INTENSITY_P:= "_atm_moon_mie_intensity"
const ATM_MOON_PARTIAL_MIE_PHASE_P:= "_atm_moon_partial_mie_phase"

# Fog
const ATM_FOG_DENSITY_P:= "_fog_density"
const ATM_FOG_RAYLEIGH_DEPTH_P:= "_fog_rayleigh_depth"
const ATM_FOG_MIE_DEPTH_P:= "_fog_mie_depth"
const ATM_FOG_FALLOFF:= "_fog_falloff"
const ATM_FOG_START:= "_fog_start"
const ATM_FOG_END:= "_fog_end"

# Near Space
const SUN_DISK_COLOR_P:= "_sun_disk_color"
const SUN_DISK_INTENSITY_P:= "_sun_disk_intensity"
const SUN_DISK_SIZE_P:= "_sun_disk_size"
const MOON_COLOR_P:= "_moon_color"
const MOON_SIZE_P:= "_moon_size"
const MOON_TEXTURE_P:= "_moon_texture"

# Deep Space
const DEEP_SPACE_MATRIX_P:= "_deep_space_matrix"
const BG_COL_P:= "_background_color"
const BG_TEXTURE_P:= "_background_texture"
const STARS_COLOR_P:= "_stars_field_color"
const STARS_TEXTURE_P:= "_stars_field_texture"
const STARS_SC_P:= "_stars_scintillation"
const STARS_SC_SPEED_P:= "_stars_scintillation_speed"

# Clouds
const CLOUDS_THICKNESS:= "_clouds_thickness"
const CLOUDS_COVERAGE:= "_clouds_coverage"
const CLOUDS_ABSORPTION:= "_clouds_absorption"
const CLOUDS_SKY_TINT_FADE:= "_clouds_sky_tint_fade"
const CLOUDS_INTENSITY:= "_clouds_intensity"
const CLOUDS_SIZE:= "_clouds_size"
const CLOUDS_NOISE_FREQ:= "_clouds_noise_freq"

const CLOUDS_UV:= "_clouds_uv"
const CLOUDS_DIRECTION:= "_clouds_direction"
const CLOUDS_SPEED:= "_clouds_speed"
const CLOUDS_TEXTURE:= "_clouds_texture"

const CLOUDS_DAY_COLOR:= "_clouds_day_color"
const CLOUDS_HORIZON_LIGHT_COLOR:= "_clouds_horizon_light_color"
const CLOUDS_NIGHT_COLOR:= "_clouds_night_color"
const CLOUDS_MIE_INTENSITY:= "_clouds_mie_intensity"
const CLOUDS_PARTIAL_MIE_PHASE:= "_clouds_partial_mie_phase"
