#pragma once

#include <vector>
#include <list>
// 这里只做前向声明，避免头文件互相包含过重
class Particle;
class Deposition;
struct WINDSGeneralData;
struct TURBGeneralData;

// 第1阶段 GPU 核心版：
// - 加入高斯扰动
// - 加入郎之万方程核心
// - 加入子步进
// - 仍未接入 interp / deposition / reflection / domainBC
void advectParticlesGPU_demo(
    double dt,
    const std::vector<Particle*>& particles,
    double boxSizeZ,
    WINDSGeneralData* WGD,
    TURBGeneralData* TGD,
    bool use_const_wind,
    double u0, double v0, double w0,
    bool clampZ_and_deactivate,
    float gaussian_scale,
    double courantNum,
    double dxy,
    double dz,
    double sim_dt,
    double vel_threshold,
    int bcTypeX,
    int bcTypeY,
    int bcTypeZ
);

// 新增：只同步 isActive / isRogue 这类轻量标志
void advectParticlesGPU_demo_syncFlagsToHost(
    const std::vector<Particle*>& particles
);

// 新增：完整同步粒子状态到 host Particle*
void advectParticlesGPU_demo_syncAllToHost(
    const std::vector<Particle*>& particles
);
void advectParticlesGPU_demo_syncOutputFieldsToHost(const std::vector<Particle*>& particles);
void advectParticlesGPU_demo_flushDepositionToHost(Deposition* deposition);
void advectParticlesGPU_demo_getLastStepStats(int &activeCount,
                                              int &inactiveCount,
                                              int &rogueCount);
void advectParticlesGPU_demo_invalidateParticleCache();
void advectParticlesGPU_demo_prepareEulerianBoxes(
    int nBoxesX, int nBoxesY, int nBoxesZ,
    float lBndx, float lBndy, float lBndz,
    float boxSizeX, float boxSizeY, float boxSizeZ);

void advectParticlesGPU_demo_accumulateEulerianBoxes(double dt);

void advectParticlesGPU_demo_flushEulerianBoxesToHost(
    std::vector<int>& pBox,
    std::vector<float>& conc,
    bool reset_after_flush);

void advectParticlesGPU_demo_resetEulerianBoxes();
// 新增：一轮模拟开始前/结束后重置 GPU cache
void advectParticlesGPU_demo_resetCache();