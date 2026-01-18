#include "presets.h"
#include <stdexcept>
#include <algorithm>

static void computeAspectResize(int inW,int inH,int targetW,int& outW,int& outH) {
  outW = targetW;
  outH = (int)((double)inH * (double)targetW / (double)inW);
  outH = std::max(1, outH);
}

GpuImage8u runPreset(const std::string& name, const GpuImage8u& inputRgb, const NppCtx& ctx) {
  if (name == "basic") {
    auto g = rgbToGray(inputRgb, ctx);
    int outW, outH; computeAspectResize(g.w, g.h, 960, outW, outH);
    auto r = resizeLinear(g, outW, outH, ctx);
    auto b = gaussianBlur(r, /*ksize*/ 9, /*sigma*/ 1.6, ctx);
    return b; // grayscale
  }

  if (name == "edges") {
    auto g = rgbToGray(inputRgb, ctx);
    auto e = sobelEdges(g, ctx);
    return e; // grayscale edges
  }

  if (name == "binary") {
    auto g = rgbToGray(inputRgb, ctx);
    auto t = thresholdBinary(g, 120, ctx);
    auto o = morphOpen3x3(t, ctx);
    return o; // binary-ish
  }

  if (name == "warp") {
    // rotate a bit + scale 0.9 + translate
    // [a0 a1 a2; b0 b1 b2]
    // This is just a sample matrix, not “correctness critical”.
    float ang = 10.0f * 3.1415926f / 180.0f;
    float s = 0.90f;
    float c = cosf(ang), si = sinf(ang);
    float a0 = s*c,  a1 = -s*si, a2 = 20.0f;
    float b0 = s*si, b1 =  s*c,  b2 = 10.0f;
    return warpAffine(inputRgb, a0,a1,a2,b0,b1,b2, inputRgb.w, inputRgb.h, ctx);
  }

  throw std::runtime_error("Unknown preset: " + name);
}
