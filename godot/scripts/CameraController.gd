extends Node3D

@export var move_speed := 5.0
@export var sprint_multiplier := 10.0
@export var mouse_sensitivity := 0.005
@export var pitch_limit := 1.5 # Radians (~85 degrees)

var yaw := 0.0
var pitch := 0.0
var mouse_locked := true

@onready var camera: Camera3D = $Camera3D

func _ready():
	var rot = rotation
	yaw = rot.y
	pitch = rot.x
	Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)

func _unhandled_input(event):
	if event.is_action_pressed("ui_cancel"): # ESC
		mouse_locked = false
		Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
	elif event is InputEventMouseButton and event.pressed:
		# Click to resume control
		mouse_locked = true
		Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)

	if event is InputEventMouseMotion and mouse_locked:
		yaw -= event.relative.x * mouse_sensitivity
		pitch = clamp(pitch - event.relative.y * mouse_sensitivity, -pitch_limit, pitch_limit)
		rotation = Vector3(pitch, yaw, 0)

func _physics_process(delta: float) -> void:
	#var frustum = camera.get_frustum();
	var direction = Vector3.ZERO

	if Input.is_key_pressed(KEY_W):
		direction -= transform.basis.z
	if Input.is_key_pressed(KEY_S):
		direction += transform.basis.z
	if Input.is_key_pressed(KEY_A):
		direction -= transform.basis.x
	if Input.is_key_pressed(KEY_D):
		direction += transform.basis.x
	if Input.is_key_pressed(KEY_Q):
		direction -= transform.basis.y
	if Input.is_key_pressed(KEY_E):
		direction += transform.basis.y
	# if Input.is_action_just_pressed("add_block"):
	# 	print("Camera position", position)
	# 	voxel_world.add_block(position, 1)

	direction = direction.normalized()

	var speed = move_speed
	if Input.is_key_pressed(KEY_SHIFT):
		speed *= sprint_multiplier

	position += direction * speed * delta
