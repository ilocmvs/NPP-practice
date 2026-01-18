#pragma once
#include <cstdint>
#include <string>
#include <vector>

struct CpuImage {
  int w = 0;
  int h = 0;
  int c = 0; // channels: 1 or 3 or 4
  std::vector<uint8_t> data; // interleaved, row-major
};

CpuImage loadImage(const std::string& path);
void saveImagePNG(const std::string& path, const CpuImage& img);
