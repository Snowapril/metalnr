// Analytic real-time baseline (docs/architecture.md §7): Beer–Lambert march with single
// scattering, multiple-scattering octaves and a two-direction ambient term.
#pragma once

#include "renderer.h"

std::unique_ptr<Renderer> makeAnalyticRenderer(MetalContext& context, const Scene& scene,
                                               const RenderConfig& config, const DensityVolume& density,
                                               std::string* error);
