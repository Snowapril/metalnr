#include "metrics.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

bool sameSize(const Image& a, const Image& b) {
    return a.width == b.width && a.height == b.height && a.width > 0 && a.height > 0 &&
           a.rgb.size() == size_t(a.width) * a.height * 3 && b.rgb.size() == a.rgb.size();
}

// Valid-region separable filter of one W x H plane with an odd-length kernel.
std::vector<double> filterValid(const std::vector<double>& in, uint32_t w, uint32_t h, const std::vector<double>& k) {
    const uint32_t r = uint32_t(k.size()) - 1;
    const uint32_t ow = w - r, oh = h - r;
    std::vector<double> tmp(size_t(ow) * h);
    for (uint32_t y = 0; y < h; ++y)
        for (uint32_t x = 0; x < ow; ++x) {
            double s = 0;
            for (uint32_t i = 0; i <= r; ++i) s += k[i] * in[size_t(y) * w + x + i];
            tmp[size_t(y) * ow + x] = s;
        }
    std::vector<double> out(size_t(ow) * oh);
    for (uint32_t y = 0; y < oh; ++y)
        for (uint32_t x = 0; x < ow; ++x) {
            double s = 0;
            for (uint32_t i = 0; i <= r; ++i) s += k[i] * tmp[size_t(y + i) * ow + x];
            out[size_t(y) * ow + x] = s;
        }
    return out;
}

// Blue -> cyan -> green -> yellow -> red for t in [0,1].
void ramp(float t, float* rgb) {
    t = std::isnan(t) ? 0.0f : std::clamp(t, 0.0f, 1.0f);
    static const float kStops[5][3] = {{0, 0, 1}, {0, 1, 1}, {0, 1, 0}, {1, 1, 0}, {1, 0, 0}};
    float x = t * 4.0f;
    int i = std::min(int(x), 3);
    float f = x - float(i);
    for (int c = 0; c < 3; ++c) rgb[c] = kStops[i][c] + f * (kStops[i + 1][c] - kStops[i][c]);
}

template <typename ErrorToRamp>
Image heatmap(const Image& test, const Image& reference, float exposure_ev, ErrorToRamp toRamp) {
    if (!sameSize(test, reference)) return Image{};
    const double s = std::exp2(double(exposure_ev));
    Image out;
    out.width = test.width;
    out.height = test.height;
    out.rgb.resize(test.rgb.size());
    for (size_t p = 0; p < size_t(test.width) * test.height; ++p) {
        double sum = 0;
        for (int c = 0; c < 3; ++c) {
            double x = s * test.rgb[p * 3 + c], r = s * reference.rgb[p * 3 + c];
            sum += toRamp(x, r);
        }
        ramp(float(sum / 3.0), &out.rgb[p * 3]);
    }
    return out;
}

}  // namespace

double relMSE(const Image& test, const Image& reference, float exposure_ev) {
    if (!sameSize(test, reference)) return kNaN;
    const double s = std::exp2(double(exposure_ev));
    double sum = 0;
    for (size_t i = 0; i < test.rgb.size(); ++i) {
        double x = s * test.rgb[i], r = s * reference.rgb[i];
        sum += (x - r) * (x - r) / (r * r + 0.01);
    }
    return sum / double(test.rgb.size());
}

double psnrDisplay(const Image& test, const Image& reference, float exposure_ev) {
    if (!sameSize(test, reference)) return kNaN;
    double sum = 0;
    for (size_t i = 0; i < test.rgb.size(); ++i) {
        double d = double(displayMap(test.rgb[i], exposure_ev)) - double(displayMap(reference.rgb[i], exposure_ev));
        sum += d * d;
    }
    double mse = sum / double(test.rgb.size());
    if (mse == 0) return std::numeric_limits<double>::infinity();
    return 10.0 * std::log10(1.0 / mse);
}

double ssimDisplay(const Image& test, const Image& reference, float exposure_ev) {
    constexpr uint32_t kSize = 11;
    constexpr double kSigma = 1.5, C1 = 0.01 * 0.01, C2 = 0.03 * 0.03;
    if (!sameSize(test, reference) || test.width < kSize || test.height < kSize) return kNaN;

    std::vector<double> k(kSize);
    double ksum = 0;
    for (uint32_t i = 0; i < kSize; ++i) {
        double d = double(i) - double(kSize / 2);
        k[i] = std::exp(-d * d / (2 * kSigma * kSigma));
        ksum += k[i];
    }
    for (double& v : k) v /= ksum;

    const uint32_t w = test.width, h = test.height;
    const size_t P = size_t(w) * h;
    double total = 0;
    size_t count = 0;
    std::vector<double> x(P), y(P), xx(P), yy(P), xy(P);
    for (int c = 0; c < 3; ++c) {
        for (size_t p = 0; p < P; ++p) {
            x[p] = displayMap(test.rgb[p * 3 + c], exposure_ev);
            y[p] = displayMap(reference.rgb[p * 3 + c], exposure_ev);
            xx[p] = x[p] * x[p];
            yy[p] = y[p] * y[p];
            xy[p] = x[p] * y[p];
        }
        std::vector<double> mx = filterValid(x, w, h, k), my = filterValid(y, w, h, k);
        std::vector<double> sxx = filterValid(xx, w, h, k), syy = filterValid(yy, w, h, k),
                            sxy = filterValid(xy, w, h, k);
        for (size_t i = 0; i < mx.size(); ++i) {
            double vx = sxx[i] - mx[i] * mx[i], vy = syy[i] - my[i] * my[i], cxy = sxy[i] - mx[i] * my[i];
            total += ((2 * mx[i] * my[i] + C1) * (2 * cxy + C2)) /
                     ((mx[i] * mx[i] + my[i] * my[i] + C1) * (vx + vy + C2));
        }
        count += mx.size();
    }
    return total / double(count);
}

Image absErrorHeatmap(const Image& test, const Image& reference, float exposure_ev, float abs_max) {
    const double scale = abs_max > 0 ? 1.0 / double(abs_max) : std::numeric_limits<double>::infinity();
    return heatmap(test, reference, exposure_ev, [scale](double x, double r) {
        double e = std::fabs(x - r);
        return e == 0 ? 0.0 : e * scale;
    });
}

Image relErrorHeatmap(const Image& test, const Image& reference, float exposure_ev) {
    return heatmap(test, reference, exposure_ev,
                   [](double x, double r) { return std::fabs(x - r) / (std::fabs(r) + 0.1); });
}
