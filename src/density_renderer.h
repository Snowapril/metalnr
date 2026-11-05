// Density visualization (milestone 1): extinction-only view of the baked volume.
#pragma once

#include "renderer.h"

std::unique_ptr<Renderer> makeDensityRenderer(MetalContext& context, const Scene& scene,
                                              const RenderConfig& config, const DensityVolume& density,
                                              std::string* error);
