// Viewer display pass: tone-maps the renderer's linear RGBA32Float output into the
// BGRA8Unorm (non-sRGB) drawable. Per channel, identical to displayMap in image_io.h:
// display = srgb_oetf(v / (1 + v)), v = max(x, 0) * 2^exposure_ev.

struct DisplayVertexOut {
    float4 position [[position]];
    float2 uv;  // (0,0) = top-left corner of the image, (1,1) = bottom-right
};

// Fullscreen triangle from vertex_id: clip-space (-1,-1), (3,-1), (-1,3).
vertex DisplayVertexOut display_vs(uint vid [[vertex_id]])
{
    float2 ndc = float2(vid == 1 ? 3.0f : -1.0f, vid == 2 ? 3.0f : -1.0f);
    DisplayVertexOut out;
    out.position = float4(ndc, 0.0f, 1.0f);
    // v flipped so row 0 of the linear texture is the top of the window.
    out.uv = float2(0.5f * ndc.x + 0.5f, 0.5f - 0.5f * ndc.y);
    return out;
}

static float display_srgb_oetf(float x)
{
    return x <= 0.0031308f ? 12.92f * x : 1.055f * pow(x, 1.0f / 2.4f) - 0.055f;
}

fragment float4 display_fs(DisplayVertexOut in [[stage_in]],
                           texture2d<float> linear_rgb [[texture(0)]],
                           constant float& exposure_ev [[buffer(0)]])
{
    // Nearest texel; exact 1:1 when the drawable has the render resolution.
    uint2 size = uint2(linear_rgb.get_width(), linear_rgb.get_height());
    uint2 p = min(uint2(in.uv * float2(size)), size - uint2(1));
    float3 v = fmax(linear_rgb.read(p).rgb, float3(0.0f)) * exp2(exposure_ev);  // fmax maps NaN to 0
    float3 t = select(v / (1.0f + v), float3(1.0f), isinf(v));
    return float4(display_srgb_oetf(t.r), display_srgb_oetf(t.g), display_srgb_oetf(t.b), 1.0f);
}
