#pragma once

#include <cuda_runtime.h>

namespace nbody
{
	/**
	 * Updates body accerations according to body positions and
	 * masses. Inspired by Lars Nyland's "Fast N-Body Simulation with
	 * CUDA".
	 * @param pos_dev  Position-and-mass vector (on device).
	 * @param acc_dev  acceration vector on (device).
	 */
	__global__ void updateAcc(const float4* pos_dev, float4* acc_dev);

	/**
	 * Updates body positions and velocities according to body
	 * accerations.
	 * @param pos_dev  Position-and-mass vector (on device).
	 * @param vel_dev  Velocity vector (on device).
	 * @param acc_dev  Acceration vector (on device).
	 * @param dt       Time increment.
	 */
	__global__ void updatePosAndVel(float4* pos_dev, float4* vel_dev,
		const float4* acc_dev, float dt);
}
