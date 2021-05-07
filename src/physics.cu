#include <device_launch_parameters.h>

#include "constants.cuh"
#include "physics.cuh"

namespace nbody
{
	__global__ void
	updateAcc(const float4* posArray_dev, float4* accArray_dev, const float* massArray_dev, size_t size)
	{
		// retrieve data for myBody (body assigned to block)
		float4 myBodyPos = posArray_dev[blockIdx.x];
		float4 myBodyAcc = { 0.0f, 0.0f, 0.0f };

		// calculate acceleration of myBody due to all other bodies
		for (unsigned int i = 0; i < size; i++) {
			float4 otherBodyPos = posArray_dev[i];
			float otherBodyMass = massArray_dev[i];

			// calculate distance vector between myBody and otherBody
			float4 dist;
			dist.x = otherBodyPos.x - myBodyPos.x;
			dist.y = otherBodyPos.y - myBodyPos.y;
			dist.z = otherBodyPos.z - myBodyPos.z;

			// calculate ratio between acceleration and distance
			float temp = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z + NAN_GUARD;
			temp = temp * temp * temp;
			float ratio = otherBodyMass / sqrtf(temp);

			// update myBody acceleration
			myBodyAcc.x += dist.x * ratio;
			myBodyAcc.y += dist.y * ratio;
			myBodyAcc.z += dist.z * ratio;
		}

		// store myBody acceleration
		accArray_dev[blockIdx.x] = myBodyAcc;
	}

	__global__ void
	updatePosAndVel(float4* posArray_dev, float4* velArray_dev, const float4* accArray_dev, float dt)
	{
		// retrieve data for myBody (body assigned to block)
		float4 myBodyPos = posArray_dev[blockIdx.x];
		float4 myBodyVel = velArray_dev[blockIdx.x];
		float4 myBodyAcc = accArray_dev[blockIdx.x];

		// update myBody position
		float dtHalfSqr = dt * dt / 2;
		myBodyPos.x += myBodyVel.x * dt + myBodyAcc.x * dtHalfSqr;
		myBodyPos.y += myBodyVel.y * dt + myBodyAcc.y * dtHalfSqr;
		myBodyPos.z += myBodyVel.z * dt + myBodyAcc.z * dtHalfSqr;

		// update myBody velocity
		myBodyVel.x += myBodyAcc.x * dt;
		myBodyVel.y += myBodyAcc.y * dt;
		myBodyVel.z += myBodyAcc.z * dt;

		// store myBody position and velocity
		posArray_dev[blockIdx.x] = myBodyPos;
		velArray_dev[blockIdx.x] = myBodyVel;
	}
}
