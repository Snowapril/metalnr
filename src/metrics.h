// Image quality metrics (docs/experiment-protocol.md §3). Host-only, no Python.
#pragma once

#include "image_io.h"

// All functions require equal dimensions; they return NaN otherwise.

// (1/3P) * sum (x - r)^2 / (r^2 + 0.01) on linear radiance scaled by 2^exposure_ev.
double relMSE(const Image& test, const Image& reference, float exposure_ev);

// PSNR in dB on displayMap()-ed images in [0,1]; +inf for identical images.
double psnrDisplay(const Image& test, const Image& reference, float exposure_ev);

// Mean SSIM on displayMap()-ed images: per channel, 11x11 Gaussian window with
// sigma 1.5, K1 = 0.01, K2 = 0.03, L = 1, valid region only (no padding),
// averaged over pixels and the 3 channels. 1.0 for identical images.
double ssimDisplay(const Image& test, const Image& reference, float exposure_ev);

// Heat maps (channel mean), mapped with a fixed blue-to-red ramp:
// absolute |x - r| scaled so that `abs_max` is red; relative |x - r| / (|r| + 0.1),
// 1.0 is red. Inputs are linear, exposure applied first.
Image absErrorHeatmap(const Image& test, const Image& reference, float exposure_ev, float abs_max);
Image relErrorHeatmap(const Image& test, const Image& reference, float exposure_ev);
