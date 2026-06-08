#version 430 core

layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aNormal;

layout(std430, binding = 0) buffer buffer0 {
    vec2 data[];
} aTexcoord;

out vec3 vertexPosition;
out vec3 vertexNormal;
out vec2 vTexcoord;

uniform mat4 model;
uniform mat4 view;
uniform mat4 projection;

uniform sampler2D normalMapSampler;

const bool enableDisplacement = true;

const float displacementScale = 0.06;

const float displacementCenter = 0.5;

void main() {
    vec2 uv = aTexcoord.data[gl_VertexID];
    uv.y = 1.0 - uv.y;
    vTexcoord = uv;

    mat3 normalMat = transpose(inverse(mat3(model)));
    vec3 worldNormal = normalize(normalMat * aNormal);

    vec4 worldPos4 = model * vec4(aPos, 1.0);
    vec3 worldPos = worldPos4.xyz / worldPos4.w;

    if (enableDisplacement) {
        float h = texture(normalMapSampler, uv).r;
        float d = (h - displacementCenter) * displacementScale;
        worldPos += worldNormal * d;
    }

    vertexPosition = worldPos;
    vertexNormal = worldNormal;

    gl_Position = projection * view * vec4(worldPos, 1.0);
}