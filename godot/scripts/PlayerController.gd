extends CharacterBody3D

@export var speed: float = 8.0
@export var sprint_speed: float = 14.0
@export var mouse_sensitivity: float = 0.15

@onready var camera_3d: Camera3D = $Camera3D
@onready var torch_light = $TorchLight

func _ready() -> void:
	Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)

func _unhandled_input(event: InputEvent) -> void:
	if event is InputEventMouseMotion and Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
		self.rotate_y(-event.relative.x * deg_to_rad(mouse_sensitivity))
		camera_3d.rotate_x(-event.relative.y * deg_to_rad(mouse_sensitivity))
		camera_3d.rotation.x = clamp(camera_3d.rotation.x, deg_to_rad(-90.0), deg_to_rad(90.0))

	if Input.is_action_pressed("ui_cancel"):
		if Input.get_mouse_mode() == Input.MOUSE_MODE_CAPTURED:
			Input.set_mouse_mode(Input.MOUSE_MODE_VISIBLE)
		else:
			Input.set_mouse_mode(Input.MOUSE_MODE_CAPTURED)
			
func _process(delta: float) -> void:
	if Input.is_action_just_pressed("toggle_torch"):
		print("toggling torch state from %s to %s" % [torch_light.visible, !torch_light.visible])
		torch_light.visible = !torch_light.visible
			
func _physics_process(delta: float) -> void:
	var input_dir := Vector3.ZERO
	
	if Input.is_key_pressed(KEY_W):
		input_dir -= transform.basis.z
	if Input.is_key_pressed(KEY_S):
		input_dir += transform.basis.z
	if Input.is_key_pressed(KEY_A):
		input_dir -= transform.basis.x
	if Input.is_key_pressed(KEY_D):
		input_dir += transform.basis.x
	
	input_dir = input_dir.normalized()
	
	var current_speed = sprint_speed if Input.is_key_pressed(KEY_SHIFT) else speed
	
	velocity.x = input_dir.x * current_speed
	velocity.z = input_dir.z * current_speed

	if Input.is_key_pressed(KEY_E):
		velocity.y = current_speed
	elif Input.is_key_pressed(KEY_Q):
		velocity.y = -current_speed
	else:
		velocity.y = 0

	move_and_slide()
