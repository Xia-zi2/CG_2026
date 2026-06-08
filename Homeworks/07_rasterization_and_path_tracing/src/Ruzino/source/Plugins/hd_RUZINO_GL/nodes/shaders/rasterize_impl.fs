#version 430

layout(location = 0) out vec3 position;
layout(location = 1) out float depth;
layout(location = 2) out vec2 texcoords;
layout(location = 3) out vec3 diffuseColor;
layout(location = 4) out vec2 metallicRoughness;
layout(location = 5) out vec3 normal;

in vec3 vertexPosition;
in vec3 vertexNormal;
in vec2 vTexcoord;

uniform mat4 projection;
uniform mat4 view;

uniform sampler2D diffuseColorSampler;
uniform sampler2D normalMapSampler;
uniform sampler2D metallicRoughnessSampler;

const bool enableDisplacement = true;

const float heightNormalStrength = 8.0;

vec3 safe_normalize(vec3 v, vec3 fallback) {
    if (dot(v, v) < 1e-12) {
        return fallback;
    }
    return normalize(v);
}

void main() {
    position = vertexPosition;

    vec4 clipPos = projection * view * vec4(position, 1.0);
    depth = clipPos.z / clipPos.w;

    texcoords = vTexcoord;
    diffuseColor = texture(diffuseColorSampler, vTexcoord).xyz;
    metallicRoughness = texture(metallicRoughnessSampler, vTexcoord).zy;

    vec3 baseNormal = safe_normalize(vertexNormal, vec3(0.0, 0.0, 1.0));

    if (!enableDisplacement) {
        vec3 normalmap_value = texture(normalMapSampler, vTexcoord).xyz;
        vec3 tangentNormal = normalize(normalmap_value * 2.0 - 1.0);

        vec3 edge1 = dFdx(vertexPosition);
        vec3 edge2 = dFdy(vertexPosition);
        vec2 deltaUV1 = dFdx(vTexcoord);
        vec2 deltaUV2 = dFdy(vTexcoord);

        vec3 tangent = edge1 * deltaUV2.y - edge2 * deltaUV1.y;

        if (length(tangent) < 1E-7) {
            vec3 bitangent_tmp = -edge1 * deltaUV2.x + edge2 * deltaUV1.x;
            tangent = normalize(cross(bitangent_tmp, baseNormal));
        }

        tangent = normalize(tangent - dot(tangent, baseNormal) * baseNormal);
        vec3 bitangent = normalize(cross(baseNormal, tangent));

        mat3 TBN = mat3(tangent, bitangent, baseNormal);
        normal = normalize(TBN * tangentNormal);
        return;
    }

    vec3 edge1 = dFdx(position);
    vec3 edge2 = dFdy(position);

    vec3 geomNormal = safe_normalize(cross(edge1, edge2), baseNormal);

    if (dot(geomNormal, baseNormal) < 0.0) {
        geomNormal = -geomNormal;
    }

    vec3 N = geomNormal;

    vec2 deltaUV1 = dFdx(vTexcoord);
    vec2 deltaUV2 = dFdy(vTexcoord);

    vec3 tangent = edge1 * deltaUV2.y - edge2 * deltaUV1.y;

    if (length(tangent) < 1E-7) {
        tangent = cross(vec3(0.0, 1.0, 0.0), N);
        if (length(tangent) < 1E-7) {
            tangent = cross(vec3(1.0, 0.0, 0.0), N);
        }
    }

    tangent = normalize(tangent - dot(tangent, N) * N);
    vec3 bitangent = normalize(cross(N, tangent));

    ivec2 texSize = textureSize(normalMapSampler, 0);
    vec2 texel = 1.0 / vec2(texSize);

    float hL = texture(normalMapSampler, vTexcoord - vec2(texel.x, 0.0)).r;
    float hR = texture(normalMapSampler, vTexcoord + vec2(texel.x, 0.0)).r;
    float hD = texture(normalMapSampler, vTexcoord - vec2(0.0, texel.y)).r;
    float hU = texture(normalMapSampler, vTexcoord + vec2(0.0, texel.y)).r;

    float dhdu = hR - hL;
    float dhdv = hU - hD;

    vec3 heightNormalTS = normalize(vec3(
        -heightNormalStrength * dhdu,
        -heightNormalStrength * dhdv,
        1.0
    ));

    mat3 TBN = mat3(tangent, bitangent, N);
    normal = normalize(TBN * heightNormalTS);
}