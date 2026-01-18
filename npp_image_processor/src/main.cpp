#include "image_io.h"
#include "npp_core.h"
#include "presets.h"
#include "utils.h"

#include <iostream>
#include <string>

static std::string argValue(int argc, char** argv, const std::string& key, const std::string& def="") {
  for (int i=1;i<argc-1;i++) if (argv[i]==key) return argv[i+1];
  return def;
}

int main(int argc, char** argv) {
  try {
    if (argc < 3) {
      std::cerr << "Usage: npp_proc <in.png> <out.png> [--preset basic|edges|binary|warp]\n";
      return 1;
    }
    std::string inPath = argv[1];
    std::string outPath = argv[2];
    std::string preset = argValue(argc, argv, "--preset", "basic");

    CpuImage input = loadImage(inPath); // RGB 8u, c=3

    // Stream (optional but good practice)
    cudaStream_t stream{};
    cudaCheck(cudaStreamCreate(&stream), "cudaStreamCreate");

    NppCtx ctx;
    ctx.stream = stream;

    // (Optional TODO for you later): set up ctx.nppStreamCtx and enable ctx.useStreamCtx.

    // Upload
    GpuImage8u d_in = gpuAlloc8u(input.w, input.h, input.c);
    uploadToGpu(input.data.data(), input.w, input.h, input.c, d_in, ctx);

    // Run preset
    GpuImage8u d_out = runPreset(preset, d_in, ctx);

    // Ensure saved as RGB
    GpuImage8u d_save = d_out;
    if (d_out.c == 1) {
      d_save = grayToRgb(d_out, ctx);
      gpuFree(d_out);
    }

    CpuImage out;
    out.w = d_save.w; out.h = d_save.h; out.c = d_save.c;
    out.data.resize((size_t)out.w*out.h*out.c);

    downloadFromGpu(d_save, out.data.data(), ctx);
    cudaCheck(cudaStreamSynchronize(stream), "stream sync");
    saveImagePNG(outPath, out);

    gpuFree(d_in);
    gpuFree(d_save);
    cudaCheck(cudaStreamDestroy(stream), "cudaStreamDestroy");

    std::cout << "Wrote: " << outPath << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 2;
  }
}
