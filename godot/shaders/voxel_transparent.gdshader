shader_type spatial;
render_mode blend_mix, depth_prepass_alpha;

uniform sampler2DArray texture_array;
uniform int atlas_columns;
uniform float ao_intensity = 0.5;

varying float layer_idx;

vec3 srgb_to_linear(vec3 color) {
    return pow(color.rgb, vec3(2.2));
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
    vec4 albedo_rgba = texture(texture_array, vec3(uv, albedo_layer));

    float normal_layer = albedo_layer + float(atlas_columns);
    vec3 normal_map_rgb = texture(texture_array, vec3(uv, normal_layer)).rgb;

    float aord_layer = albedo_layer + float(2 * atlas_columns);
    vec3 aord_rgb = texture(texture_array, vec3(uv, aord_layer)).rgb;

    ALBEDO = srgb_to_linear(albedo_rgba.rgb);
    ALBEDO *= mix(1.0, pow(aord_rgb.r, 2.0), ao_intensity);
    NORMAL = apply_normal_map(TANGENT, BINORMAL, NORMAL, normal_map_rgb);
    ROUGHNESS = aord_rgb.g;

    ALPHA = 0.15;
    //ALPHA = albedo_rgba.a;
}
