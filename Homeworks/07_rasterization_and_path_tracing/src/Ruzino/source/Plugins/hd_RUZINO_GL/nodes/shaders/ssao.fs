#version 430 core

uniform vec2 iResolution;

uniform sampler2D positionSampler;
uniform sampler2D depthSampler;
uniform sampler2D normalSampler;

uniform mat4 view;
uniform mat4 projection;

layout(location = 0) out vec4 Color;

const int SAMPLE_COUNT = 32;

const float SSAO_RADIUS   = 2.00;
const float SSAO_STRENGTH = 6.00;
const float AO_POWER      = 1.30;
const float SSAO_BIAS     = 0.02;

const float PI = 3.14159265358979323846;

bool insideUV(vec2 uv)
{
    return uv.x >= 0.0 && uv.x <= 1.0 &&
           uv.y >= 0.0 && uv.y <= 1.0;
}

float hash12(vec2 p)
{
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

float viewDepth(vec3 worldPos)
{
    vec4 viewPos = view * vec4(worldPos, 1.0);
    return -viewPos.z;
}

vec3 fallbackTangent(vec3 n)
{
    vec3 up = abs(n.z) < 0.999
        ? vec3(0.0, 0.0, 1.0)
        : vec3(0.0, 1.0, 0.0);

    return normalize(cross(up, n));
}

mat3 buildTBN(vec3 n, vec2 uv)
{
    float r1 = hash12(uv * iResolution + vec2(12.9898, 78.233));
    float r2 = hash12(uv * iResolution + vec2(39.3468, 11.135));
    float r3 = hash12(uv * iResolution + vec2(73.1562, 52.235));

    vec3 randomVec = normalize(vec3(
        r1 * 2.0 - 1.0,
        r2 * 2.0 - 1.0,
        r3 * 2.0 - 1.0
    ));

    vec3 tangent = randomVec - n * dot(randomVec, n);

    if (length(tangent) < 1e-5) {
        tangent = fallbackTangent(n);
    } else {
        tangent = normalize(tangent);
    }

    vec3 bitangent = normalize(cross(n, tangent));

    return mat3(tangent, bitangent, n);
}

vec3 hemisphereSample(int i, vec2 uv)
{
    float fi = float(i);

    float u1 = fract((fi + 0.5) * 0.754877666 +
                     hash12(uv * iResolution + vec2(17.0, 29.0)));

    float u2 = fract((fi + 0.5) * 0.569840296 +
                     hash12(uv * iResolution + vec2(43.0, 61.0)));

    float phi = 2.0 * PI * u1;

    /*
        cosine-weighted hemisphere:
        x = sqrt(u2) cos(phi)
        y = sqrt(u2) sin(phi)
        z = sqrt(1 - u2)
    */
    float r = sqrt(u2);
    float z = sqrt(max(0.0, 1.0 - u2));

    vec3 sampleDir = vec3(
        r * cos(phi),
        r * sin(phi),
        z
    );
    float scale = fi / float(SAMPLE_COUNT);
    scale = mix(0.10, 1.00, scale * scale);

    return sampleDir * scale;
}

void main()
{
    vec2 uv = gl_FragCoord.xy / iResolution;

    vec3 p = texture(positionSampler, uv).xyz;
    vec3 nRaw = texture(normalSampler, uv).xyz;
    if (length(nRaw) < 1e-5) {
        Color = vec4(vec3(1.0), 1.0);
        return;
    }

    vec3 n = normalize(nRaw);
    mat3 TBN = buildTBN(n, uv);

    float occlusion = 0.0;
    float validCount = 0.0;

    for (int i = 0; i < SAMPLE_COUNT; ++i) {
        vec3 sampleLocal = hemisphereSample(i, uv);

        vec3 sampleWorld =
            p + TBN * sampleLocal * SSAO_RADIUS;

        vec4 sampleClip =
            projection * view * vec4(sampleWorld, 1.0);

        if (sampleClip.w <= 0.0) {
            continue;
        }

        vec3 sampleNDC = sampleClip.xyz / sampleClip.w;
        vec2 sampleUV = sampleNDC.xy * 0.5 + 0.5;

        if (!insideUV(sampleUV)) {
            continue;
        }

        vec3 q = texture(positionSampler, sampleUV).xyz;
        vec3 qNormal = texture(normalSampler, sampleUV).xyz;

        if (length(qNormal) < 1e-5) {
            continue;
        }

        vec3 v = q - p;
        float worldDist = length(v);

        if (worldDist < 1e-5 || worldDist > SSAO_RADIUS * 2.0) {
            continue;
        }

        vec3 dir = v / worldDist;

        float hemi = max(dot(n, dir) - SSAO_BIAS, 0.0);

        float sampleDepth = viewDepth(sampleWorld);
        float bufferDepth = viewDepth(q);
        float depthDiff = abs(bufferDepth - sampleDepth);

        float depthWeight =
            1.0 - smoothstep(0.0, SSAO_RADIUS, depthDiff);

        float distWeight =
            1.0 - smoothstep(0.0, SSAO_RADIUS * 2.0, worldDist);

        float contribution = hemi * depthWeight * distWeight;

        occlusion += contribution;
        validCount += 1.0;
    }

    float ao = 1.0;

    if (validCount > 0.0) {
        ao = 1.0 - SSAO_STRENGTH * occlusion / validCount;
    }

    ao = clamp(ao, 0.0, 1.0);
    ao = pow(ao, AO_POWER);

    Color = vec4(vec3(ao), 1.0);
}