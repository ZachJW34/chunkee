extends CharacterBody3D

@export var speed: float = 5.0
@export var sprint_speed: float = 8.0
@export var jump_velocity: float = 7.0
@export var mouse_sensitivity: float = 0.15

@export var air_acceleration: float = 10.0

@export var gravity: float = -20.0

@onready var camera_3d: Camera3D = $Camera3D
@onready var torch_light = $TorchLight


func _ready() -> void:
	Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)


func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
		self.rotate_y(-event.relative.x * deg_to_rad(mouse_sensitivity))
		camera_3d.rotate_x(-event.relative.y * deg_to_rad(mouse_sensitivity))
		camera_3d.rotation.x = clamp(camera_3d.rotation.x, deg_to_rad(-90.0), deg_to_rad(90.0))

	if Input.is_action_just_pressed("ui_cancel"):
		if Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
			Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
		else:
			Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)


func _process(delta: float) -> void:
	if Input.is_action_just_pressed("toggle_torch"):
		torch_light.visible = !torch_light.visible


func _physics_process(delta: float) -> void:
	if not is_on_floor():
		velocity.y += gravity * delta

	if Input.is_action_pressed("jump") and is_on_floor():
		velocity.y = jump_velocity

	var current_speed = sprint_speed if Input.is_action_pressed("sprint") else speed
	var input_dir := Vector3.ZERO
	if Input.is_action_pressed("move_forward"):
		input_dir -= transform.basis.z
	if Input.is_action_pressed("move_backward"):
		input_dir += transform.basis.z
	if Input.is_action_pressed("move_left"):
		input_dir -= transform.basis.x
	if Input.is_action_pressed("move_right"):
		input_dir += transform.basis.x
	input_dir = input_dir.normalized()

	if is_on_floor():
		velocity.x = input_dir.x * current_speed
		velocity.z = input_dir.z * current_speed
	else:
		velocity.x = move_toward(velocity.x, input_dir.x * current_speed, air_acceleration * delta)
		velocity.z = move_toward(velocity.z, input_dir.z * current_speed, air_acceleration * delta)

	move_and_slide()
