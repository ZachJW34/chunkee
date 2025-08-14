extends CharacterBody3D

## Walking Movement Variables
@export_group("Walking")
@export var speed: float = 5.0
@export var sprint_speed: float = 8.0
@export var jump_velocity: float = 7.0
@export var air_acceleration: float = 10.0
@export var gravity: float = -20.0
@export var max_velocity: float = 50.0

## Flying Movement Variables
@export_group("Flying")
@export var fly_speed: float = 12.0 # Speed for flying mode

## General Variables
@export_group("General")
@export var mouse_sensitivity: float = 0.15

@onready var camera_3d: Camera3D = $Camera3D
@onready var torch_light = $TorchLight

# State variable to track fly mode
var is_flying: bool = false


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
	# Toggle fly mode when the action is pressed
	if Input.is_action_just_pressed("toggle_fly"):
		is_flying = !is_flying
		velocity = Vector3.ZERO # Reset velocity to prevent momentum carry-over

	# Choose movement logic based on the current state
	if is_flying:
		_handle_fly_movement(delta)
	else:
		_handle_walk_movement(delta)

	move_and_slide()


# Handles the original walking and jumping logic
func _handle_walk_movement(delta: float) -> void:
	# Apply gravity only when not on the floor
	if not is_on_floor():
		velocity.y += gravity * delta

	# Handle jump
	if Input.is_action_pressed("jump") and is_on_floor():
		velocity.y = jump_velocity

	# Determine current speed (sprint or normal)
	var current_speed = sprint_speed if Input.is_action_pressed("sprint") else speed
	
	# Get input direction from player
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

	# Apply movement based on whether the character is on the floor or in the air
	if is_on_floor():
		velocity.x = input_dir.x * current_speed
		velocity.z = input_dir.z * current_speed
	else:
		velocity.x = move_toward(velocity.x, input_dir.x * current_speed, air_acceleration * delta)
		velocity.z = move_toward(velocity.z, input_dir.z * current_speed, air_acceleration * delta)
		
	# Clamp velocity to a maximum value
	if velocity.length() > max_velocity:
		velocity = velocity.normalized() * max_velocity


# Handles the new flying logic
func _handle_fly_movement(delta: float) -> void:
	# Use fly_speed, but allow sprinting to go faster
	var current_speed = fly_speed * 1.5 if Input.is_action_pressed("sprint") else fly_speed

	# Get input direction based on the CAMERA's GLOBAL orientation
	var input_dir := Vector3.ZERO
	if Input.is_action_pressed("move_forward"):
		input_dir -= camera_3d.global_transform.basis.z
	if Input.is_action_pressed("move_backward"):
		input_dir += camera_3d.global_transform.basis.z
	if Input.is_action_pressed("move_left"):
		input_dir -= camera_3d.global_transform.basis.x
	if Input.is_action_pressed("move_right"):
		input_dir += camera_3d.global_transform.basis.x

	input_dir = input_dir.normalized()
	
	# Apply velocity directly for responsive flight
	velocity = input_dir * current_speed
