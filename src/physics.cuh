#ifndef _PHYSICS_CUH_
#define _PHYSICS_CUH_

#include <cuda_runtime.h>

/**
 * Updates acceleration vector according to position and mass vectors.
 *
 * @param pos_dev   Position vector (on device).
 * @param acc_dev   Acceleration vector (on device).
 * @param mass_dev  Mass vector (on device).
 * @param n         Vector size.
 */
__global__ void
update_acc(const float3 *pos_dev, float3 *acc_dev, const float *mass_dev, unsigned int n);

/**
 * Updates position and velocity vectors according to acceleration vector.
 *
 * @param pos_dev  Position vector (on device).
 * @param vel_dev  Velocity vector (on device).
 * @param acc_dev  Acceleration vector (on device).
 */
__global__ void
update_pos_and_vel(float3 *pos_dev, float3 *vel_dev, const float3 *acc_dev);

#endif
