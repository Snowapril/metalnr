// Reference path tracer (docs/architecture.md §6), registry name "pt".
// Progressive: each encode() adds one sample per pixel to a running mean.
// stats(): bounce_cap_hits, mean_bounces, paths (since the last reset).
#pragma once

#include "renderer.h"

#include <cstdint>

// Safety cap on real collisions per path; the only bias source of the reference.
constexpr uint32_t kPathTracerBounceCap = 4096;

std::unique_ptr<Renderer> makePathTracerRenderer(MetalContext& context, const Scene& scene,
                                                 const RenderConfig& config, const DensityVolume& density,
                                                 std::string* error);
