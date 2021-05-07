#pragma once

#include <cuda_runtime.h>

namespace nbody
{
	/**
	 * Updates body accelerations according to body positions and masses.
	 * @param posArray_dev   Body positions array on device.
	 * @param accArray_dev   Body accelerations array on device.
	 * @param massArray_dev  Body masses array on device.
	 * @param size           Size of arrays.
	 */
	__global__ void
	updateAcc(const float4* posArray_dev, float4* accArray_dev, const float* massArray_dev, size_t size);

	/**
	 * Updates body positions and velocities according to body accelerations.
	 * @param posArray_dev  Body positions array on device.
	 * @param velArray_dev  Body velocities array on device.
	 * @param accArray_dev  Body accelerations array on device.
	 * @param dt            Time increment.
	 */
	__global__ void
	updatePosAndVel(float4* posArray_dev, float4* velArray_dev, const float4* accArray_dev, float dt);
}
