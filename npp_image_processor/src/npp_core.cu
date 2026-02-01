#include "npp_core.h"
#include "utils.h"
#include <stdexcept>
#include <vector>
#include <cmath>
#include <cstdint>
#include <algorithm>

static NppiSize sz(int w, int h) { return NppiSize{w,h}; }
static NppiRect roi(int w, int h) { return NppiRect{0,0,w,h}; }

GpuImage8u gpuAlloc8u(int w, int h, int c) {
  GpuImage8u out;
  out.w=w; out.h=h; out.c=c;
  size_t pitchBytes = 0;

  // TODO: use cudaMallocPitch for out.d and set out.pitch
  // pitchBytes should be in bytes; width in bytes is w*c*sizeof(Npp8u)
  cudaMallocPitch((void**)&out.d, &pitchBytes, w*c*sizeof(Npp8u), h);
  out.pitch = static_cast<int>(pitchBytes);
  return out;
}

GpuImage32f gpuAlloc32f(int w, int h, int c) {
  GpuImage32f out;
  out.w=w; out.h=h; out.c=c;
  size_t pitchBytes = 0;

  // TODO: cudaMallocPitch for Npp32f; width in bytes is w*c*sizeof(Npp32f)
  cudaMallocPitch((void**)&out.d, &pitchBytes, w*c*sizeof(Npp32f), h);
  out.pitch = static_cast<int>(pitchBytes);

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
  cudaMemcpy2DAsync(dst.d, dst.pitch, h_data, w*c*sizeof(Npp8u), w*c*sizeof(Npp8u), h, cudaMemcpyHostToDevice, ctx.stream);
  (void)ctx;
}

void downloadFromGpu(const GpuImage8u& src, uint8_t* h_out, const NppCtx& ctx) {
  // TODO: cudaMemcpy2DAsync from src.d to h_out
  cudaMemcpy2DAsync(h_out, src.w*src.c*sizeof(Npp8u), src.d, src.pitch, src.w*src.c*sizeof(Npp8u), src.h, cudaMemcpyDeviceToHost, ctx.stream);
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
  nppiRGBToGray_8u_C3C1R_Ctx(rgb.d, rgb.pitch, out.d, out.pitch, sz(rgb.w, rgb.h), ctx.nppStreamCtx);

  (void)ctx;
  return out;
}

GpuImage8u grayToRgb(const GpuImage8u& gray, const NppCtx& ctx) {
  if (gray.c != 1) throw std::runtime_error("grayToRgb expects c=1");
  GpuImage8u out = gpuAlloc8u(gray.w, gray.h, 3);

  // TODO: nppiGrayToRGB_8u_C1C3R
  NppiSize roi = { gray.w, gray.h };
  nppiDup_8u_C1C3R_Ctx(gray.d, gray.pitch, out.d, out.pitch, roi, ctx.nppStreamCtx);

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
  if (src.c == 1) {
    nppiResize_8u_C1R_Ctx(src.d, src.pitch, sz(src.w, src.h), roi(src.w, src.h),
                          out.d, out.pitch, sz(outW, outH), roi(outW, outH),
                           NPPI_INTER_LINEAR, ctx.nppStreamCtx);
  } else if (src.c == 3) {
    nppiResize_8u_C3R_Ctx(src.d, src.pitch, sz(src.w, src.h), roi(src.w, src.h),
                          out.d, out.pitch, sz(outW, outH), roi(outW, outH),
                           NPPI_INTER_LINEAR, ctx.nppStreamCtx);
  } else {
    throw std::runtime_error("resizeLinear: unsupported channel count");
  }

  (void)ctx;
  return out;
}

//2D Gaussian kernel builder (integer quantized)
// static void buildGaussianKernel2D_Int(
//     int ksize, double sigma,
//     std::vector<Npp32s>& kernel,
//     int& divisor,
//     int scaleBits = 14 // scale ~ 16384, good precision
// ){
//   if (ksize % 2 == 0 || ksize < 3) throw std::runtime_error("ksize must be odd >=3");
//   int r = ksize / 2;

//   std::vector<double> w(ksize * ksize);
//   double sum = 0.0;

//   double s2 = 2.0 * sigma * sigma;
//   for (int y = -r; y <= r; ++y) {
//     for (int x = -r; x <= r; ++x) {
//       double val = std::exp(-(x*x + y*y) / s2);
//       w[(y + r)*ksize + (x + r)] = val;
//       sum += val;
//     }
//   }

//   // normalize
//   for (double& v : w) v /= sum;

//   // quantize
//   const int scale = 1 << scaleBits;
//   kernel.resize(ksize * ksize);

//   int div = 0;
//   for (int i = 0; i < ksize*ksize; ++i) {
//     int q = (int)std::llround(w[i] * scale);
//     kernel[i] = (Npp32s)q;
//     div += q;
//   }

//   // Guard: divisor must be nonzero
//   if (div == 0) {
//     // fallback: put 1 in center
//     std::fill(kernel.begin(), kernel.end(), 0);
//     kernel[(r*ksize + r)] = 1;
//     div = 1;
//   }
//   divisor = div;
// }

GpuImage8u gaussianBlur(const GpuImage8u& srcGray, int ksize, double sigma, const NppCtx& ctx) {
  if (srcGray.c != 1) throw std::runtime_error("gaussianBlur expects grayscale");
  GpuImage8u out = gpuAlloc8u(srcGray.w, srcGray.h, 1);

  // TODO: Use an NPP filtering path.
  // Options depending on NPP version:
  // - nppiFilterGaussian_8u_C1R (if available)
  // - or build a Gaussian kernel and use nppiFilter_8u_C1R
  // Practice goal: allocate kernel on host, pass to NPP filter.
  nppiFilterGauss_8u_C1R_Ctx(srcGray.d, srcGray.pitch, out.d, out.pitch, sz(srcGray.w, srcGray.h), 
                              (ksize == 3) ? NPP_MASK_SIZE_3_X_3 : 
                              (ksize == 5) ? NPP_MASK_SIZE_5_X_5 :
                              (ksize == 7) ? NPP_MASK_SIZE_7_X_7 :
                              (ksize == 9) ? NPP_MASK_SIZE_9_X_9 :
                              (ksize ==11) ? NPP_MASK_SIZE_11_X_11 :
                              NPP_MASK_SIZE_13_X_13,
                              ctx.nppStreamCtx);
  // std::vector<Npp32s> kernel;
  // int divisor = 0;
  // buildGaussianKernel2D_Int(ksize, sigma, kernel, divisor);
  // NppStatus status = nppiFilter_8u_C1R(
  //     srcGray.d, srcGray.pitch,
  //     out.d, out.pitch,
  //     sz(srcGray.w, srcGray.h),
  //     kernel.data(),
  //     sz(ksize, ksize),
  //     sz(ksize/2, ksize/2),
  //     divisor
  // );
  (void)ksize; (void)sigma; (void)ctx;
  return out;
}

__global__ void magKernel(const float* gx, int gxPitchFloats,
                          const float* gy, int gyPitchFloats,
                          float* mag, int magPitchFloats,
                          int w, int h, float scale)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float dx = gx[y * gxPitchFloats + x];
    float dy = gy[y * gyPitchFloats + x];
    float m = sqrtf(dx*dx + dy*dy) * scale;
    mag[y * magPitchFloats + x] = m;
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
  GpuImage32f tmp32f_image = gpuAlloc32f(srcGray.w, srcGray.h, 1);
  nppiConvert_8u32f_C1R_Ctx(
      srcGray.d, srcGray.pitch,
      tmp32f_image.d, tmp32f_image.pitch,
      sz(tmp32f_image.w, tmp32f_image.h),
      ctx.nppStreamCtx
  );

  GpuImage32f gradX = gpuAlloc32f(srcGray.w, srcGray.h, 1);
  nppiFilterSobelHoriz_32f_C1R_Ctx(
      tmp32f_image.d, tmp32f_image.pitch,
      gradX.d, gradX.pitch,
      sz(gradX.w, gradX.h),
      ctx.nppStreamCtx
  );

  GpuImage32f gradY = gpuAlloc32f(srcGray.w, srcGray.h, 1);
  nppiFilterSobelVert_32f_C1R_Ctx(
      tmp32f_image.d, tmp32f_image.pitch,
      gradY.d, gradY.pitch,
      sz(gradY.w, gradY.h),
      ctx.nppStreamCtx
  );

  GpuImage32f mag = gpuAlloc32f(srcGray.w, srcGray.h, 1);
  int gxpitchFloats = gradX.pitch / sizeof(Npp32f);
  int gypitchFloats = gradY.pitch / sizeof(Npp32f);
  int magpitchFloats = mag.pitch / sizeof(Npp32f);
  dim3 blockSize(16,16);
  dim3 gridSize( (srcGray.w + blockSize.x - 1)/blockSize.x,
                 (srcGray.h + blockSize.y - 1)/blockSize.y );
  float scale = 1.0f; // adjust as needed
  magKernel<<<gridSize, blockSize, 0, ctx.stream>>>(
      gradX.d, gxpitchFloats,
      gradY.d, gypitchFloats,
      mag.d, magpitchFloats,
      srcGray.w, srcGray.h,
      scale
  );

  GpuImage8u out = gpuAlloc8u(srcGray.w, srcGray.h, 1);
  nppiConvert_32f8u_C1R_Ctx(
      mag.d, mag.pitch,
      out.d, out.pitch,
      sz(srcGray.w, srcGray.h),
      NPP_RND_NEAR,
      ctx.nppStreamCtx
  );

  (void)ctx;
  return out;
}

GpuImage8u thresholdBinary(const GpuImage8u& srcGray, uint8_t thresh, const NppCtx& ctx) {
  if (srcGray.c != 1) throw std::runtime_error("thresholdBinary expects grayscale");
  GpuImage8u out = gpuAlloc8u(srcGray.w, srcGray.h, 1);

  // TODO: nppiThreshold_8u_C1R (or compare against constant)
  // Ensure output is 0/255.
  nppiThreshold_8u_C1R_Ctx(
      srcGray.d, srcGray.pitch,
      out.d, out.pitch,
      sz(srcGray.w, srcGray.h),
      thresh,
      NPP_CMP_LESS
      , ctx.nppStreamCtx
  );

  (void)thresh; (void)ctx;
  return out;
}

GpuImage8u morphOpen3x3(const GpuImage8u& srcBin, const NppCtx& ctx) {
  if (srcBin.c != 1) throw std::runtime_error("morphOpen3x3 expects C1");

  GpuImage8u tmp = gpuAlloc8u(srcBin.w, srcBin.h, 1);
  GpuImage8u out = gpuAlloc8u(srcBin.w, srcBin.h, 1);
  NppStatus st;

  NppiSize roi = {srcBin.w, srcBin.h};

  st = nppiErode3x3_8u_C1R_Ctx(
      srcBin.d, srcBin.pitch,
      tmp.d, tmp.pitch,
      roi,
      ctx.nppStreamCtx);
  if (st != NPP_SUCCESS) throw std::runtime_error("nppiErode failed");

  st = nppiDilate3x3_8u_C1R_Ctx(
      tmp.d, tmp.pitch,
      out.d, out.pitch,
      roi,
      ctx.nppStreamCtx);
  if (st != NPP_SUCCESS) throw std::runtime_error("nppiDilate failed");

  gpuFree(tmp);
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
  double affineMatrix[2][3] = {
      {static_cast<double>(a0), static_cast<double>(a1), static_cast<double>(a2)},
      {static_cast<double>(b0), static_cast<double>(b1), static_cast<double>(b2)}
  };
  if (src.c == 1) {
    nppiWarpAffine_8u_C1R_Ctx(
        src.d, 
        sz(src.w, src.h),
        src.pitch,
        roi(src.w, src.h),
        out.d, out.pitch,
        roi(outW, outH),
        affineMatrix,
        NPPI_INTER_LINEAR,
        ctx.nppStreamCtx
    );
  } else if (src.c == 3) {
    nppiWarpAffine_8u_C3R_Ctx(
        src.d,
        sz(src.w, src.h),
        src.pitch,
        roi(src.w, src.h),
        out.d, out.pitch,
        roi(outW, outH),
        affineMatrix,
        NPPI_INTER_LINEAR,
        ctx.nppStreamCtx
    );
  } else {
    throw std::runtime_error("warpAffine: unsupported channel count");
  }

  (void)a0;(void)a1;(void)a2;(void)b0;(void)b1;(void)b2;(void)ctx;
  return out;
}
