#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "image_io.h"
#include <stdexcept>

CpuImage loadImage(const std::string& path) {
  CpuImage out;
  int w,h,c;
  // Load as 3 channels (RGB) to keep NPP paths simpler
  uint8_t* p = stbi_load(path.c_str(), &w, &h, &c, 3);
  if (!p) throw std::runtime_error("stbi_load failed");
  out.w = w; out.h = h; out.c = 3;
  out.data.assign(p, p + (size_t)w*h*3);
  stbi_image_free(p);
  return out;
}

void saveImagePNG(const std::string& path, const CpuImage& img) {
  if (img.data.empty()) throw std::runtime_error("saveImagePNG: empty image");
  int stride = img.w * img.c;
  if (!stbi_write_png(path.c_str(), img.w, img.h, img.c, img.data.data(), stride)) {
    throw std::runtime_error("stbi_write_png failed");
  }
}
