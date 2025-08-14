shader_type spatial;
// Using blend_mix for transparency and depth_prepass for sorting
render_mode blend_mix, depth_prepass_alpha, cull_back;

// --- Uniforms for Standard Voxel Blocks ---
uniform sampler2DArray texture_array;
uniform int atlas_columns;
uniform float ao_intensity = 0.5;

// --- Uniforms for Water ---
uniform vec4 water_color : source_color = vec4(0.1, 0.4, 0.7, 0.5);
uniform float water_roughness : hint_range(0.0, 1.0) = 0.05;
uniform sampler2D noise_texture;
uniform float wave_strength = 0.05;
uniform float wave_speed = 0.2;
uniform vec2 wave_direction = vec2(1.0, 0.0);
uniform float refraction_strength = 0.02;
uniform float fresnel_power = 2.0;
uniform float ripple_strength = 0.5;
uniform sampler2D screen_texture : hint_screen_texture, repeat_disable, filter_linear_mipmap;

// This varying carries the texture ID from the vertex to the fragment shader
varying float layer_idx;

// --- Helper Functions ---
vec3 srgb_to_linear(vec3 color) {
	return pow(color.rgb, vec3(2.2));
}

vec3 apply_normal_map(vec3 tangent, vec3 binormal, vec3 normal, vec3 normal_map_sample) {
	vec3 n = normalize(normal_map_sample * 2.0 - 1.0);
	mat3 tbn = mat3(normalize(tangent), normalize(binormal), normalize(normal));
	return normalize(tbn * n);
}

void vertex() {
	// Get the layer index from the mesh's custom data
	layer_idx = CUSTOM0.x;

	// If the block is water (layer 4), animate its vertices to create waves
	if (layer_idx == 999.0) {
		vec2 time_offset = (wave_direction * TIME * wave_speed);
		// Sample noise using world position for seamless waves across blocks
		float noise = texture(noise_texture, VERTEX.xz * 0.1 + time_offset).r;
		VERTEX.y += noise * wave_strength;
	}
}

void fragment() {
	// Check if the current fragment is water
	if (layer_idx == 999.0) {
    // --- 💧 Water Rendering Logic ---
	    vec2 time_offset1 = (wave_direction * TIME * wave_speed);
	    vec2 time_offset2 = (vec2(0.5, 1.0) * TIME * wave_speed * 0.7);

	    float noise1 = texture(noise_texture, UV * 2.0 + time_offset1).r;
	    float noise2 = texture(noise_texture, UV * 1.0 + time_offset2).r;
	    float combined_noise = (noise1 + noise2) * 0.5;

	    // Perturb normals to create ripples for lighting and refraction
	    // We create a new normal by adding the noise derivatives to the base normal.
	    // This is a fast way to simulate ripples on a flat surface.
	    vec3 perturbed_normal = normalize(NORMAL + vec3(dFdx(combined_noise), 0.0, dFdy(combined_noise)) * ripple_strength);
	    NORMAL = perturbed_normal; // CORRECTED: Assign to NORMAL

	    // Distort SCREEN_UV with normals to create refraction
	    vec2 refraction_uv = SCREEN_UV + (perturbed_normal.xy * refraction_strength);
	    vec4 refracted_color = texture(screen_texture, refraction_uv);

	    // Mix the water color with the refracted view from behind
	    ALBEDO = mix(refracted_color.rgb, water_color.rgb, water_color.a);

	    // Use Fresnel to make water more reflective at shallow angles
	    float fresnel = pow(1.0 - dot(VIEW, NORMAL), fresnel_power);
	    ALPHA = water_color.a + fresnel * (1.0 - water_color.a);

	    ROUGHNESS = water_roughness; // Use a dedicated roughness for water
	    METALLIC = 0.2;

	} else {
		// --- 🧱 Standard Block Rendering Logic ---
		vec2 uv = UV;
		float albedo_layer = float(layer_idx);
		vec4 albedo_rgba = texture(texture_array, vec3(uv, albedo_layer));

		float normal_layer = albedo_layer + float(atlas_columns);
		vec3 normal_map_rgb = texture(texture_array, vec3(uv, normal_layer)).rgb;

		float aord_layer = albedo_layer + float(2 * atlas_columns);
		vec3 aord_rgb = texture(texture_array, vec3(uv, aord_layer)).rgb;

		ALBEDO = srgb_to_linear(albedo_rgba.rgb);
		ALBEDO *= mix(1.0, pow(aord_rgb.r, 2.0), ao_intensity);
		NORMAL = apply_normal_map(TANGENT, BINORMAL, NORMAL, normal_map_rgb);
		ROUGHNESS = aord_rgb.g;
		ALPHA = 0.25; // Standard blocks are fully opaque
	}
}
