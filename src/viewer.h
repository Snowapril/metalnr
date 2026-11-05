// Interactive window (AppKit + CAMetalLayer). Blocks until the window closes.
#pragma once

#include "scene.h"

#include <string>

class MetalContext;

// Opens a window of scene.width x scene.height points, renders `config.renderer`
// every display frame and presents it through a tone-mapping render pass.
// Controls: left-drag orbit around the camera target, scroll / trackpad pinch
// zoom (distance to target), W/A/S/D pan the target in the view plane, Q/E move
// it down/up, '[' / ']' exposure -/+ 0.5 EV, R reset to the scene camera,
// 1-4 switch renderer (density, analytic, pt, neural) when available.
// Returns a process exit code; *error set when nonzero.
// autotestPrefix non-empty: scripted self-test driven by frame count through the same
// control functions as user input (orbit, zoom, pan, exposure, reset). Each step writes
// <prefix>_<name>.png from the presented drawable and prints one `autotest <name> ...`
// line; the window then closes itself and `autotest done gpu_errors=<n>` is printed.
int runViewer(MetalContext& context, const Scene& scene, const std::string& cameraName,
              const RenderConfig& config, std::string* error, const std::string& autotestPrefix = "");
