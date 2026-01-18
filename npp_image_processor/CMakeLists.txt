cmake_minimum_required(VERSION 3.20)
project(npp_image_processor LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

find_package(CUDAToolkit REQUIRED)

add_executable(npp_proc
  src/main.cpp
  src/image_io.cpp
  src/presets.cpp
  src/npp_core.cu
)

target_include_directories(npp_proc PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${CMAKE_CURRENT_SOURCE_DIR}/external
)

# NPP libs are split. We link the common ones you’ll likely need.
target_link_libraries(npp_proc PRIVATE
  CUDA::cudart
  CUDA::nppc
  CUDA::nppi
  CUDA::nppif
)

# Useful for debugging device errors
if (MSVC)
  target_compile_options(npp_proc PRIVATE /W4)
else()
  target_compile_options(npp_proc PRIVATE -Wall -Wextra -Wpedantic)
endif()