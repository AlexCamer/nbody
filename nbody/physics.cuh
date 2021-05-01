#ifndef _PHYSICS_CUH_
#define _PHYSICS_CUH_

#include <cuda_runtime.h>

__global__ void update_acc(const float3* pos_dev, float3* acc_dev, const float* mass_dev, unsigned int n);
__global__ void update_pos_and_vel(float3* pos_dev, float3* vel_dev, const float3* acc_dev);

#endif
