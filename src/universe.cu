#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "physics.cuh"
#include "universe.h"

namespace nbody
{
	struct Universe::Data
	{
		// host memory
		thrust::host_vector<float4> posVec_host;  // body positions vector on host
		thrust::host_vector<float4> velVec_host;  // body velocities vector on host
		thrust::host_vector<float4> accVec_host;  // body accelerations vector on host
		thrust::host_vector<float> massVec_host;  // body masses vector on host
	};

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
		data->posVec_host.push_back({ pos[0], pos[1], pos[2] });
		data->velVec_host.push_back({ vel[0], vel[1], vel[2] });
		data->accVec_host.push_back({});
		data->massVec_host.push_back(mass);
	}

	void
	Universe::remove(unsigned int bodyIdx)
	{
		data->posVec_host.erase(data->posVec_host.begin() + bodyIdx);
		data->velVec_host.erase(data->velVec_host.begin() + bodyIdx);
		data->accVec_host.erase(data->accVec_host.begin() + bodyIdx);
		data->massVec_host.erase(data->massVec_host.begin() + bodyIdx);
	}

	void
	Universe::reset()
	{
		data->posVec_host.clear();
		data->velVec_host.clear();
		data->accVec_host.clear();
		data->massVec_host.clear();
	}

	size_t
	Universe::size()
	{
		return data->posVec_host.size();
	}

	void
	Universe::step(float dt)
	{
		float4* posArray_host = thrust::raw_pointer_cast(data->posVec_host.data());
		float4* velArray_host = thrust::raw_pointer_cast(data->velVec_host.data());
		float4* accArray_host = thrust::raw_pointer_cast(data->accVec_host.data());
		float* massArray_host = thrust::raw_pointer_cast(data->massVec_host.data());

		for (int i = 0; i < size(); i++) {
			// update accelerations based on positions and masses
			updateAcc(posArray_host, accArray_host, massArray_host, size(), i);
		}

		for (int i = 0; i < size(); i++) {
			// update positions and velocities based on new accelerations
			updatePosAndVel(posArray_host, velArray_host, accArray_host, dt, i);
		}
	}

	const float*
	Universe::position(unsigned int bodyIdx)
	{
		// return body position array (form: [x, y, z])
		return (float*)&data->posVec_host[bodyIdx];
	}

	const float*
	Universe::velocity(unsigned int bodyIdx)
	{
		// return body velocity array (form: [x, y, z])
		return (float*)&data->velVec_host[bodyIdx];
	}

	float
	Universe::mass(unsigned int bodyIdx)
	{
		// return body mass
		return data->massVec_host[bodyIdx];
	}
}
