#pragma once
#include <cstdint>
#include <cuda_runtime.h>

// NPP headers
#include <npp.h>
#include <nppi.h>

struct GpuImage8u {
  int w = 0, h = 0, c = 0;   // channels
  int pitch = 0;            // bytes per row
  Npp8u* d = nullptr;       // device pointer (pitched)
};

struct GpuImage32f {
  int w = 0, h = 0, c = 0;
  int pitch = 0;            // bytes per row
  Npp32f* d = nullptr;
};

struct NppCtx {
  cudaStream_t stream = nullptr;
  // Optional: NPP stream context (lets NPP use your stream reliably)
  NppStreamContext nppStreamCtx{};
  bool useStreamCtx = false;
};

GpuImage8u  gpuAlloc8u(int w, int h, int c);
GpuImage32f gpuAlloc32f(int w, int h, int c);
void gpuFree(GpuImage8u& img);
void gpuFree(GpuImage32f& img);

void uploadToGpu(const uint8_t* h_data, int w, int h, int c, GpuImage8u& dst, const NppCtx& ctx);
void downloadFromGpu(const GpuImage8u& src, uint8_t* h_out, const NppCtx& ctx);

// --- Common NPP ops (implement via NPP calls) ---
GpuImage8u rgbToGray(const GpuImage8u& rgb, const NppCtx& ctx);            // 3->1
GpuImage8u grayToRgb(const GpuImage8u& gray, const NppCtx& ctx);           // 1->3 (for saving)
GpuImage8u resizeLinear(const GpuImage8u& src, int outW, int outH, const NppCtx& ctx);
GpuImage8u gaussianBlur(const GpuImage8u& srcGray, int ksize, double sigma, const NppCtx& ctx);
GpuImage8u sobelEdges(const GpuImage8u& srcGray, const NppCtx& ctx);
GpuImage8u thresholdBinary(const GpuImage8u& srcGray, uint8_t thresh, const NppCtx& ctx);
GpuImage8u morphOpen3x3(const GpuImage8u& srcBin, const NppCtx& ctx);
GpuImage8u warpAffine(const GpuImage8u& src, float a0,float a1,float a2,float b0,float b1,float b2,
                      int outW,int outH, const NppCtx& ctx);
