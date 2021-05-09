#include <device_launch_parameters.h>

#include "constants.h"
#include "physics.cuh"

namespace nbody
{
	__device__ float4 accFromOther(float4 myPos, float4 otherPos,
		float4 myAcc)
	{
		// calculate distance between my body and other body
		float4 dist;
		dist.x = otherPos.x - myPos.x;
		dist.y = otherPos.y - myPos.y;
		dist.z = otherPos.z - myPos.z;

		// calculate ratio between acceration and distance
		// NOTE: pos.w stores body mass for faster read
		float distSqr = dist.x * dist.x + dist.y * dist.y
			+ dist.z * dist.z + NAN_GUARD;
		float distSixth = distSqr * distSqr * distSqr;
		float distRatio = otherPos.w * rsqrt(distSixth);

		// update myBody acceration
		myAcc.x += dist.x * distRatio;
		myAcc.y += dist.y * distRatio;
		myAcc.z += dist.z * distRatio;

		return myAcc;
	}

	__global__ void updateAcc(const float4* pos_dev, float4* acc_dev)
	{
		// position-and-mass cache
		extern __shared__ float4 sharedPos[];

		// retrieve data for myBody (body assigned to block)
		float4 myPos = pos_dev[blockIdx.x];
		float4 myAcc = { 0.0f, 0.0f, 0.0f };

		// calculate acceration of my body due to all other bodies
		for (int tileIdx = 0; tileIdx < gridDim.x; tileIdx++) {
			// populate cache (each thread contributes one position)
			int otherIdx = tileIdx * blockDim.x + threadIdx.x;
			sharedPos[threadIdx.x] = pos_dev[otherIdx];

			// synchronize to keep cache data relevant
			__syncthreads();
			for (otherIdx = 0; otherIdx < blockDim.x; otherIdx++) {
				float4 otherPos = sharedPos[otherIdx];
				myAcc = accFromOther(myPos, otherPos, myAcc);
			}
			__syncthreads();
		}

		// store myBody acceration
		acc_dev[blockIdx.x] = myAcc;
	}

	__global__ void updatePosAndVel(float4* pos_dev, float4* vel_dev,
		const float4* acc_dev, float dt)
	{
		// retrieve data for myBody (body assigned to block)
		float4 myBodyPos = pos_dev[blockIdx.x];
		float4 myBodyVel = vel_dev[blockIdx.x];
		float4 myBodyAcc = acc_dev[blockIdx.x];

		// update myBody velocity
		myBodyVel.x += myBodyAcc.x * dt;
		myBodyVel.y += myBodyAcc.y * dt;
		myBodyVel.z += myBodyAcc.z * dt;

		// update myBody position
		myBodyPos.x += myBodyVel.x * dt;
		myBodyPos.y += myBodyVel.y * dt;
		myBodyPos.z += myBodyVel.z * dt;

		// store myBody position and velocity
		pos_dev[blockIdx.x] = myBodyPos;
		vel_dev[blockIdx.x] = myBodyVel;
	}
}
