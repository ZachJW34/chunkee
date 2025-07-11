shader_type spatial;
render_mode blend_mix, depth_draw_opaque;

uniform sampler2DArray texture_array;
uniform int atlas_columns = 4;
uniform float ao_intensity = 0.5;

varying float layer_idx;

vec3 srgb_to_linear(vec3 color) {
    return pow(color, vec3(2.2));
}

void vertex() {
    layer_idx = CUSTOM0.x;
}

vec3 apply_normal_map(vec3 tangent, vec3 binormal, vec3 normal, vec3 normal_map_sample) {
    vec3 n = normalize(normal_map_sample * 2.0 - 1.0);
    mat3 tbn = mat3(normalize(tangent), normalize(binormal), normalize(normal));
    return normalize(tbn * n);
}

void fragment() {
    vec2 uv = UV;

    float albedo_layer = float(layer_idx);
    float normal_layer = albedo_layer + float(atlas_columns);
    float specular_layer = albedo_layer + float(2 * atlas_columns);

    // albedo (row 1)
    vec3 albedo_rgb = texture(texture_array, vec3(uv, albedo_layer)).rgb;

    // normal + AO + height (row 2)
    vec4 normal_rgba = texture(texture_array, vec3(uv, normal_layer));
    vec3 normal_map_rgb = normal_rgba.rgb;
    float ao = 1.0 - normal_rgba.b; // LabPBR blue is AO (0=occluded)

    // specular (row 3)
    vec4 specular_rgba = texture(texture_array, vec3(uv, specular_layer));
    float perceptual_smoothness = specular_rgba.r;
    float roughness = pow(1.0 - perceptual_smoothness, 2.0);
    // you *could* also get:
    // float f0 = specular_rgba.g;
    // float porosity = specular_rgba.b;
    // float emissive = specular_rgba.a;

    ALBEDO = srgb_to_linear(albedo_rgb) * mix(1.0, ao, ao_intensity);
    NORMAL = apply_normal_map(TANGENT, BINORMAL, NORMAL, normal_map_rgb);
    ROUGHNESS = roughness;

    // optionally:
    // SPECULAR = f0;
    // EMISSION = srgb_to_linear(albedo_rgb) * emissive;
}
