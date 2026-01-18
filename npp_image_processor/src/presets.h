#pragma once
#include <string>
#include "npp_core.h"

GpuImage8u runPreset(const std::string& name, const GpuImage8u& inputRgb, const NppCtx& ctx);
