#pragma once

#include <vector>

class WINDSGeneralData;

// Stage-1 GPU offload for LocalMixingSerial:
// only accelerates the dominant "wall below" contribution.
// Returns true on success; caller should fall back to CPU on false.
bool localMixingWallBelowGPU(
    WINDSGeneralData *WGD,
    int max_height,
    const std::vector<float> &x_cc,
    const std::vector<float> &y_cc,
    const std::vector<float> &z_cc,
    const std::vector<float> &z_fc,
    const std::vector<int> &wall_below_work);
