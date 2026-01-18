#pragma once
#include <stdexcept>
#include <string>
#include <cuda_runtime.h>

inline void cudaCheck(cudaError_t e, const char* msg) {
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + ": " + cudaGetErrorString(e));
  }
}

inline void cudaSyncCheck(const char* msg) {
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + " (last error): " + cudaGetErrorString(e));
  }
  e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    throw std::runtime_error(std::string(msg) + " (sync): " + cudaGetErrorString(e));
  }
}
