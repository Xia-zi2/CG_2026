#include "path.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

#include "../surfaceInteraction.h"
RUZINO_NAMESPACE_OPEN_SCOPE
using namespace pxr;

VtValue PathIntegrator::Li(const GfRay& ray, std::default_random_engine& random)
{
    std::uniform_real_distribution<float> uniform_dist(
        0.0f, 1.0f - std::numeric_limits<float>::epsilon());
    std::function<float()> uniform_float = std::bind(uniform_dist, random);

    auto color = EstimateOutGoingRadiance(ray, uniform_float, 0);

    return VtValue(GfVec3f(color[0], color[1], color[2]));
}

GfVec3f PathIntegrator::EstimateOutGoingRadiance(
    const GfRay& ray,
    const std::function<float()>& uniform_float,
    int recursion_depth)
{
    const int maxDepth = 50;
    const int rrStartDepth = 3;
    const float rrProb = 0.8f;
    const float eps = 1e-6f;

    if (recursion_depth >= maxDepth) {
        return GfVec3f(0.0f);
    }

    SurfaceInteraction si;

    if (!Intersect(ray, si)) {
        if (recursion_depth == 0) {
            GfVec3f lightPos{ 0.0f };
            return IntersectLights(ray, lightPos);
        }

        return GfVec3f(0.0f);
    }

    if (recursion_depth == 0) {
        GfVec3f lightPos{ 0.0f };
        GfVec3f lightRadiance = IntersectLights(ray, lightPos);

        bool hasLightRadiance = lightRadiance[0] > 0.0f ||
                                lightRadiance[1] > 0.0f ||
                                lightRadiance[2] > 0.0f;

        if (hasLightRadiance) {
            GfVec3f origin = GfVec3f(ray.GetStartPoint());

            float lightDist2 = GfDot(lightPos - origin, lightPos - origin);
            float surfaceDist2 =
                GfDot(si.position - origin, si.position - origin);

            if (lightDist2 < surfaceDist2) {
                return lightRadiance;
            }
        }
    }

    if (GfDot(si.shadingNormal, ray.GetDirection()) > 0) {
        si.flipNormal();
        si.PrepareTransforms();
    }

    GfVec3f directLight = EstimateDirectLight(si, uniform_float);

    float continueProb = 1.0f;
    if (recursion_depth >= rrStartDepth) {
        continueProb = rrProb;
        if (uniform_float() > continueProb) {
            return directLight;
        }
    }

    GfVec3f wi;
    float brdfPdf = 0.0f;

    GfVec3f brdfVal = si.Sample(wi, brdfPdf, uniform_float);

    if (brdfPdf <= eps) {
        return directLight;
    }

    wi.Normalize();

    float cosTheta = GfDot(si.shadingNormal, wi);
    if (cosTheta <= eps) {
        return directLight;
    }

    GfVec3f offsetNormal = (GfDot(wi, si.geometricNormal) >= 0.0f)
                               ? si.geometricNormal
                               : -si.geometricNormal;

    GfVec3f nextOrigin = si.position + 0.0001f * offsetNormal;

    GfRay nextRay(
        GfVec3d(nextOrigin[0], nextOrigin[1], nextOrigin[2]),
        GfVec3d(wi[0], wi[1], wi[2]));

    GfVec3f incoming =
        EstimateOutGoingRadiance(nextRay, uniform_float, recursion_depth + 1);

    GfVec3f globalLight =
        GfCompMult(brdfVal, incoming) * cosTheta / (brdfPdf * continueProb);

    return directLight + globalLight;
}

RUZINO_NAMESPACE_CLOSE_SCOPE
