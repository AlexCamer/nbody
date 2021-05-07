#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "physics.cuh"
#include "universe.h"

namespace nbody
{
	struct Universe::Data
	{
		// device memory
		thrust::device_vector<float4> posVec_dev;  // body positions vector on device
		thrust::device_vector<float4> velVec_dev;  // body velocities vector on device
		thrust::device_vector<float4> accVec_dev;  // body accelerations vector on device
		thrust::device_vector<float> massVec_dev;  // body masses vector on device

		// host memory
		thrust::host_vector<float4> posVec_host;  // body positions vector on host
		thrust::host_vector<float4> velVec_host;  // body velocities vector on host
		thrust::host_vector<float> massVec_host;  // body masses vector on host

		// flags
		bool isPosVecOnHostValid;   // is body positions vector on host up to date?
		bool isVelVecOnHostValid;   // is body velocities vector on host up to date?
		bool isMassVecOnHostValid;  // is body masses vector on host up to date?

		/**
		 * Sets flags to false.
		 */
		void invalidate();
	};

	void
	Universe::Data::invalidate()
	{
		isPosVecOnHostValid = false;
		isVelVecOnHostValid = false;
		isMassVecOnHostValid = false;
	}

	Universe::Universe() : data{ new Data{} }
	{
	}

	Universe::~Universe()
	{
		delete data;
	}

	void
	Universe::add(const float* pos, const float* vel, float mass)
	{
		data->posVec_dev.push_back({ pos[0], pos[1], pos[2] });
		data->velVec_dev.push_back({ vel[0], vel[1], vel[2] });
		data->accVec_dev.push_back({});
		data->massVec_dev.push_back(mass);

		// vectors on device were changed
		data->invalidate();
	}

	void
	Universe::remove(unsigned int bodyIdx)
	{
		data->posVec_dev.erase(data->posVec_dev.begin() + bodyIdx);
		data->velVec_dev.erase(data->velVec_dev.begin() + bodyIdx);
		data->accVec_dev.erase(data->accVec_dev.begin() + bodyIdx);
		data->massVec_dev.erase(data->massVec_dev.begin() + bodyIdx);

		// vectors on device were changed
		data->invalidate();
	}

	void
	Universe::reset()
	{
		data->posVec_dev.clear();
		data->velVec_dev.clear();
		data->accVec_dev.clear();
		data->massVec_dev.clear();

		// vectors on device were changed
		data->invalidate();
	}

	size_t
	Universe::size()
	{
		return data->posVec_dev.size();
	}

	void
	Universe::step(float dt)
	{
		float4* posArray_dev = thrust::raw_pointer_cast(data->posVec_dev.data());
		float4* velArray_dev = thrust::raw_pointer_cast(data->velVec_dev.data());
		float4* accArray_dev = thrust::raw_pointer_cast(data->accVec_dev.data());
		float* massArray_dev = thrust::raw_pointer_cast(data->massVec_dev.data());
		unsigned int numBlocks = (unsigned int)size();

		// update accelerations based on positions and masses
		updateAcc<<<numBlocks, 1>>>(posArray_dev, accArray_dev, massArray_dev, size());

		// wait for device to finish computation
		cudaDeviceSynchronize();

		// update positions and velocities based on new accelerations
		updatePosAndVel<<<numBlocks, 1>>>(posArray_dev, velArray_dev, accArray_dev, dt);

		// wait for device to finish computation
		cudaDeviceSynchronize();

		// vectors on device were changed
		data->invalidate();
	}

	const float*
	Universe::position(unsigned int bodyIdx)
	{
		// if body positions vector on host is invalid:
		if (!data->isPosVecOnHostValid) {
			// copy from body positions vector on device
			data->posVec_host = data->posVec_dev;  // abstracted device to host memcopy
			data->isPosVecOnHostValid = true;
		}

		// return body position array (form: [x, y, z])
		return (float*)&data->posVec_host[bodyIdx];
	}

	const float*
	Universe::velocity(unsigned int bodyIdx)
	{
		// if body velocities vector on host is invalid:
		if (!data->isVelVecOnHostValid) {
			// copy from body velocities vector on device
			data->velVec_host = data->velVec_dev;  // abstracted device to host memcopy
			data->isVelVecOnHostValid = true;
		}

		// return body velocity array (form: [x, y, z])
		return (float*)&data->velVec_host[bodyIdx];
	}

	float
	Universe::mass(unsigned int bodyIdx)
	{
		// if body masses vector on host is invalid:
		if (!data->isMassVecOnHostValid) {
			// copy from body masses vector on device
			data->massVec_host = data->massVec_dev;  // abstracted device to host memcopy
			data->isMassVecOnHostValid = true;
		}

		// return body mass
		return data->massVec_host[bodyIdx];
	}
}
