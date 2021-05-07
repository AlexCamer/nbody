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
	void
	updateAcc(const float4* posArray_host, float4* accArray_host, const float* massArray_host,
		size_t size, unsigned int bodyIdx);

	/**
	 * Updates body positions and velocities according to body accelerations.
	 * @param posArray_dev  Body positions array on device.
	 * @param velArray_dev  Body velocities array on device.
	 * @param accArray_dev  Body accelerations array on device.
	 * @param dt            Time increment.
	 */
	void
	updatePosAndVel(float4* posArray_host, float4* velArray_host, const float4* accArray_host,
		float dt, unsigned int bodyIdx);
}
