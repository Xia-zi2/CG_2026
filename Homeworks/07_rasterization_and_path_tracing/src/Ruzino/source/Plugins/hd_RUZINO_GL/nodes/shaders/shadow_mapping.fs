#version 430 core

uniform mat4 light_view;
uniform mat4 light_projection;

in vec3 vertexPosition;

layout(location = 0) out float shadow_map0;

const float LIGHT_NEAR = 1.0;
const float LIGHT_FAR  = 25.0;

void main() {
    vec4 lightViewPos = light_view * vec4(vertexPosition, 1.0);

    float zLight = -lightViewPos.z;

    shadow_map0 = clamp(
        (zLight - LIGHT_NEAR) / (LIGHT_FAR - LIGHT_NEAR),
        0.0,
        1.0
    );
}