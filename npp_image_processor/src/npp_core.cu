#include "npp_core.h"
#include "utils.h"
#include <stdexcept>
#include <vector>
#include <cmath>

static NppiSize sz(int w, int h) { return NppiSize{w,h}; }
static NppiRect roi(int w, int h) { return NppiRect{0,0,w,h}; }

GpuImage8u gpuAlloc8u(int w, int h, int c) {
  GpuImage8u out;
  out.w=w; out.h=h; out.c=c;
  size_t pitchBytes = 0;

  // TODO: use cudaMallocPitch for out.d and set out.pitch
  // pitchBytes should be in bytes; width in bytes is w*c*sizeof(Npp8u)

  return out;
}

GpuImage32f gpuAlloc32f(int w, int h, int c) {
  GpuImage32f out;
  out.w=w; out.h=h; out.c=c;
  size_t pitchBytes = 0;

  // TODO: cudaMallocPitch for Npp32f; width in bytes is w*c*sizeof(Npp32f)

  return out;
}

void gpuFree(GpuImage8u& img) {
  if (img.d) cudaFree(img.d);
  img = {};
}

void gpuFree(GpuImage32f& img) {
  if (img.d) cudaFree(img.d);
  img = {};
}

void uploadToGpu(const uint8_t* h_data, int w, int h, int c, GpuImage8u& dst, const NppCtx& ctx) {
  if (!dst.d || dst.w!=w || dst.h!=h || dst.c!=c) {
    throw std::runtime_error("uploadToGpu: dst not allocated or wrong shape");
  }
  // TODO: cudaMemcpy2DAsync into dst.d with dst.pitch
  // src pitch = w*c bytes
  (void)ctx;
}

void downloadFromGpu(const GpuImage8u& src, uint8_t* h_out, const NppCtx& ctx) {
  // TODO: cudaMemcpy2DAsync from src.d to h_out
  (void)ctx;
}

// ------------- NPP ops -------------
GpuImage8u rgbToGray(const GpuImage8u& rgb, const NppCtx& ctx) {
  if (rgb.c != 3) throw std::runtime_error("rgbToGray expects c=3");
  GpuImage8u out = gpuAlloc8u(rgb.w, rgb.h, 1);

  // TODO: call nppiRGBToGray_8u_C3C1R (or BGR variant depending on your data)
  // - src pointer: rgb.d
  // - src step: rgb.pitch
  // - dst pointer: out.d
  // - dst step: out.pitch
  // - image size: sz(rgb.w, rgb.h)
  // If using stream ctx: use *Ctx variants if available in your installed NPP.

  (void)ctx;
  return out;
}

GpuImage8u grayToRgb(const GpuImage8u& gray, const NppCtx& ctx) {
  if (gray.c != 1) throw std::runtime_error("grayToRgb expects c=1");
  GpuImage8u out = gpuAlloc8u(gray.w, gray.h, 3);

  // TODO: nppiGrayToRGB_8u_C1C3R

  (void)ctx;
  return out;
}

GpuImage8u resizeLinear(const GpuImage8u& src, int outW, int outH, const NppCtx& ctx) {
  GpuImage8u out = gpuAlloc8u(outW, outH, src.c);

  // TODO: nppiResize_8u_CxR (choose C1R/C3R based on src.c)
  // Typical parameters include:
  // - src ROI rect, dst ROI rect
  // - scale factors or computed from sizes
  // Use NPPI_INTER_LINEAR

  (void)ctx;
  return out;
}

GpuImage8u gaussianBlur(const GpuImage8u& srcGray, int ksize, double sigma, const NppCtx& ctx) {
  if (srcGray.c != 1) throw std::runtime_error("gaussianBlur expects grayscale");
  GpuImage8u out = gpuAlloc8u(srcGray.w, srcGray.h, 1);

  // TODO: Use an NPP filtering path.
  // Options depending on NPP version:
  // - nppiFilterGaussian_8u_C1R (if available)
  // - or build a Gaussian kernel and use nppiFilter_8u_C1R
  // Practice goal: allocate kernel on host, pass to NPP filter.

  (void)ksize; (void)sigma; (void)ctx;
  return out;
}

GpuImage8u sobelEdges(const GpuImage8u& srcGray, const NppCtx& ctx) {
  if (srcGray.c != 1) throw std::runtime_error("sobelEdges expects grayscale");

  // Output can be 8u for simplicity, but Sobel typically produces signed / wider range.
  // Common practice: compute into 16s or 32f, then convert back.
  // We'll guide you into a 16s path then scale to 8u.

  // TODO:
  // 1) alloc GpuImage16s-like (you can add a struct, or reuse 32f)
  // 2) nppiFilterSobelHoriz/Vert (or generic Sobel) into 16s/32f
  // 3) magnitude (either NPP or a tiny custom kernel)
  // 4) convert/scale back to 8u out

  (void)ctx;
  return gpuAlloc8u(srcGray.w, srcGray.h, 1); // placeholder; replace
}

GpuImage8u thresholdBinary(const GpuImage8u& srcGray, uint8_t thresh, const NppCtx& ctx) {
  if (srcGray.c != 1) throw std::runtime_error("thresholdBinary expects grayscale");
  GpuImage8u out = gpuAlloc8u(srcGray.w, srcGray.h, 1);

  // TODO: nppiThreshold_8u_C1R (or compare against constant)
  // Ensure output is 0/255.

  (void)thresh; (void)ctx;
  return out;
}

GpuImage8u morphOpen3x3(const GpuImage8u& srcBin, const NppCtx& ctx) {
  if (srcBin.c != 1) throw std::runtime_error("morphOpen3x3 expects 1 channel");
  GpuImage8u out = gpuAlloc8u(srcBin.w, srcBin.h, 1);

  // TODO:
  // opening = erode then dilate, 3x3 mask
  // Use NPP morphology functions available in your version:
  // - nppiErode_8u_C1R / nppiDilate_8u_C1R
  // Many variants require an “NPP buffer” size query + temp buffer allocation.
  // This is good practice: call the *GetBufferSize function and allocate scratch.

  (void)ctx;
  return out;
}

GpuImage8u warpAffine(const GpuImage8u& src,
                      float a0,float a1,float a2,float b0,float b1,float b2,
                      int outW,int outH, const NppCtx& ctx) {
  GpuImage8u out = gpuAlloc8u(outW, outH, src.c);

  // TODO: nppiWarpAffine_8u_CxR (C1/C3)
  // Matrix usually passed as double[2][3] or similar depending on API.
  // Set interpolation: NPPI_INTER_LINEAR
  // Define src ROI and dst ROI

  (void)a0;(void)a1;(void)a2;(void)b0;(void)b1;(void)b2;(void)ctx;
  return out;
}
