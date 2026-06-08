#version 430 core

struct Light {
    mat4 light_projection;
    mat4 light_view;
    vec3 position;
    float radius;
    vec3 color;
    int shadow_map_id;
};

layout(std430, binding = 0) buffer lightsBuffer {
    Light lights[];
};

uniform vec2 iResolution;

uniform sampler2D diffuseColorSampler;
uniform sampler2D normalMapSampler;
uniform sampler2D metallicRoughnessSampler;
uniform sampler2DArray shadow_maps;
uniform sampler2D position;
uniform sampler2D ambientOcclusionSampler;

uniform vec3 camPos;
uniform int light_count;

layout(location = 0) out vec4 Color;

const float LIGHT_NEAR = 1.0;
const float LIGHT_FAR  = 25.0;
const float LIGHT_FOVY = 2.09439510239;

const int BLOCKER_SAMPLES = 32;
const int PCF_SAMPLES = 32;
const float PCSS_RADIUS_SCALE = 0.07;

const vec2 poissonDisk32[32] = vec2[](
    vec2( 0.04529686,  0.11650405),
    vec2(-0.19415918, -0.09579778),
    vec2( 0.26836857, -0.07812368),
    vec2(-0.17170220,  0.28265412),
    vec2(-0.07293458, -0.36783902),
    vec2( 0.33415147,  0.24539314),
    vec2(-0.44805818,  0.04867101),
    vec2( 0.31957418, -0.36365828),
    vec2( 0.01064963,  0.51527816),
    vec2(-0.37627245, -0.39407365),
    vec2( 0.57154162,  0.03827760),
    vec2(-0.46810849,  0.37449892),
    vec2( 0.09612332, -0.61756401),
    vec2( 0.35986489,  0.54071458),
    vec2(-0.65353901, -0.16128164),
    vec2( 0.61087776, -0.33347168),

    vec2(-0.23233588,  0.67944465),
    vec2(-0.29622920, -0.67758635),
    vec2( 0.69518293,  0.30797028),
    vec2(-0.73985756,  0.24896947),
    vec2( 0.38692670, -0.70065165),
    vec2( 0.19250586,  0.79675372),
    vec2(-0.69578485, -0.46798338),
    vec2( 0.84739398, -0.12766535),
    vec2(-0.54994555,  0.68057688),
    vec2(-0.05530562, -0.89096369),
    vec2( 0.65509715,  0.63164288),
    vec2(-0.92672241, -0.02367651),
    vec2( 0.71193106, -0.61949913),
    vec2(-0.10834230,  0.95401098),
    vec2(-0.57402560, -0.78969590),
    vec2( 0.97225765,  0.19771206)
);

bool insideUV(vec2 uv) {
    return uv.x >= 0.0 && uv.x <= 1.0 &&
           uv.y >= 0.0 && uv.y <= 1.0;
}

float decodeLinearDepth(float depth01) {
    return depth01 * (LIGHT_FAR - LIGHT_NEAR) + LIGHT_NEAR;
}

float lightWidthUV(float lightRadius) {
    float nearPlaneHeight =
        2.0 * LIGHT_NEAR * tan(0.5 * LIGHT_FOVY);

    float lightWidth =
        2.0 * lightRadius * PCSS_RADIUS_SCALE;

    return lightWidth / nearPlaneHeight;
}

float blockerSearchRadiusUV(float lightRadius, float zReceiver) {
    float wLightUV = lightWidthUV(lightRadius);

    return wLightUV *
           (zReceiver - LIGHT_NEAR) /
           max(zReceiver, 1e-4);
}

float findAverageBlockerDepth(
    vec2 shadowUV,
    float zReceiver,
    int layer,
    float searchRadiusUV,
    float bias,
    out int blockerCount
) {
    float blockerDepthSum = 0.0;
    blockerCount = 0;

    if (insideUV(shadowUV)) {
        float centerDepth01 =
            texture(shadow_maps, vec3(shadowUV, float(layer))).x;

        float centerDepth =
            decodeLinearDepth(centerDepth01);

        if (zReceiver - bias > centerDepth) {
            blockerDepthSum += centerDepth;
            blockerCount++;
        }
    }

    for (int k = 0; k < BLOCKER_SAMPLES; ++k) {
        vec2 sampleUV =
            shadowUV + poissonDisk32[k] * searchRadiusUV;

        if (!insideUV(sampleUV)) {
            continue;
        }

        float sampleDepth01 =
            texture(shadow_maps, vec3(sampleUV, float(layer))).x;

        float sampleDepth =
            decodeLinearDepth(sampleDepth01);

        if (zReceiver - bias > sampleDepth) {
            blockerDepthSum += sampleDepth;
            blockerCount++;
        }
    }

    if (blockerCount == 0) {
        return -1.0;
    }

    return blockerDepthSum / float(blockerCount);
}

float penumbraRadiusUV(
    float lightRadius,
    float zReceiver,
    float zBlocker
) {
    float penumbraRatio =
        (zReceiver - zBlocker) / max(zBlocker, 1e-4);

    float wLightUV = lightWidthUV(lightRadius);

    return penumbraRatio *
           wLightUV *
           LIGHT_NEAR /
           max(zReceiver, 1e-4);
}

float pcfFilter(
    vec2 shadowUV,
    float zReceiver,
    int layer,
    float filterRadiusUV,
    float bias
) {
    float shadowSum = 0.0;
    int validCount = 0;

    if (insideUV(shadowUV)) {
        float centerDepth01 =
            texture(shadow_maps, vec3(shadowUV, float(layer))).x;

        float centerDepth =
            decodeLinearDepth(centerDepth01);

        shadowSum +=
            (zReceiver - bias > centerDepth) ? 1.0 : 0.0;

        validCount++;
    }

    for (int k = 0; k < PCF_SAMPLES; ++k) {
        vec2 sampleUV =
            shadowUV + poissonDisk32[k] * filterRadiusUV;

        if (!insideUV(sampleUV)) {
            continue;
        }

        float sampleDepth01 =
            texture(shadow_maps, vec3(sampleUV, float(layer))).x;

        float sampleDepth =
            decodeLinearDepth(sampleDepth01);

        shadowSum +=
            (zReceiver - bias > sampleDepth) ? 1.0 : 0.0;

        validCount++;
    }

    if (validCount == 0) {
        return 0.0;
    }

    return shadowSum / float(validCount);
}

float pcssShadow(
    vec2 shadowUV,
    float zReceiver,
    int layer,
    float lightRadius,
    float bias
) {
    float searchRadiusUV =
        blockerSearchRadiusUV(lightRadius, zReceiver);

    int blockerCount = 0;
    float avgBlockerDepth =
        findAverageBlockerDepth(
            shadowUV,
            zReceiver,
            layer,
            searchRadiusUV,
            bias,
            blockerCount
        );

    if (blockerCount == 0) {
        return 0.0;
    }

    float filterRadiusUV =
        penumbraRadiusUV(
            lightRadius,
            zReceiver,
            avgBlockerDepth
        );

    vec2 shadowMapSize = vec2(textureSize(shadow_maps, 0).xy);
    float texelSize =
        1.0 / min(shadowMapSize.x, shadowMapSize.y);

    filterRadiusUV = clamp(filterRadiusUV, texelSize, 0.10);

    return pcfFilter(
        shadowUV,
        zReceiver,
        layer,
        filterRadiusUV,
        bias
    );
}

void main() {
    vec2 uv = gl_FragCoord.xy / iResolution;

    vec3 pos = texture(position, uv).xyz;
    vec3 normal = texture(normalMapSampler, uv).xyz;

    vec4 metalnessRoughness = texture(metallicRoughnessSampler, uv);
    float metal = clamp(metalnessRoughness.x, 0.0, 1.0);
    float roughness = clamp(metalnessRoughness.y, 0.0, 1.0);

    vec3 baseColor = texture(diffuseColorSampler, uv).xyz;
    normal = normalize(normal);

    float ao = texture(ambientOcclusionSampler, uv).r;
    ao = clamp(ao, 0.0, 1.0);

    vec3 dielectricSpecular = vec3(0.04);
    vec3 kd = baseColor * (1.0 - dielectricSpecular.r) * (1.0 - metal);
    vec3 ks = mix(dielectricSpecular, baseColor, metal);

    float ambientStrength = 1;
    float shininess = mix(128.0, 8.0, clamp(roughness, 0.0, 1.0));

    vec3 viewDir = normalize(camPos - pos);

    Color = vec4(ambientStrength * kd * ao, 1.0);

    for (int i = 0; i < light_count; i++) {
        vec3 lightDir = normalize(lights[i].position - pos);
        vec3 halfDir = normalize(lightDir + viewDir);

        float diff = max(dot(normal, lightDir), 0.0);

        float spec = 0.0;
        if (diff > 0.0) {
            spec = pow(max(dot(normal, halfDir), 0.0), shininess);
        }

        vec3 lightColor = lights[i].color;

        vec3 diffuse = lightColor * kd * diff;
        vec3 specular = lightColor * ks * spec;

        float visibility = 1.0;

        if (diff > 0.0) {
            vec4 lightViewPos =
                lights[i].light_view * vec4(pos, 1.0);

            vec4 lightClip =
                lights[i].light_projection * lightViewPos;

            if (lightClip.w > 0.0) {
                vec3 lightNDC = lightClip.xyz / lightClip.w;

                bool insideShadowFrustum =
                    lightNDC.x >= -1.0 && lightNDC.x <= 1.0 &&
                    lightNDC.y >= -1.0 && lightNDC.y <= 1.0 &&
                    lightNDC.z >= -1.0 && lightNDC.z <= 1.0;

                if (insideShadowFrustum) {
                    vec2 shadowUV = lightNDC.xy * 0.5 + 0.5;

                    float zReceiver = -lightViewPos.z;

                    float ndotl = max(dot(normal, lightDir), 0.0);

                    float bias = max(0.015 * (1.0 - ndotl), 0.06);

                    float shadow = pcssShadow(
                        shadowUV,
                        zReceiver,
                        lights[i].shadow_map_id,
                        lights[i].radius,
                        bias
                    );

                    visibility = 1.0 - shadow;
                }
            }
        }

        Color += vec4(visibility * (diffuse + specular), 0.0);
    }
}