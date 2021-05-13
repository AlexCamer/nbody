#include <device_launch_parameters.h>

#include "constants.h"
#include "physics.cuh"

namespace nbody
{
	__device__ float4 accFromOther(float4 myPos, float4 otherPos,
		float4 myAcc)
	{
		// calculate distance between my-body and other-body
		float4 dist;
		dist.x = otherPos.x - myPos.x;
		dist.y = otherPos.y - myPos.y;
		dist.z = otherPos.z - myPos.z;

		// calculate ratio between acceleration and distance
		// NOTE: pos.w stores body mass for faster read
		float distSqr = dist.x * dist.x + dist.y * dist.y
			+ dist.z * dist.z + NAN_GUARD;
		float distSixth = distSqr * distSqr * distSqr;
		float ratio = otherPos.w * rsqrt(distSixth);

		// update my-body acceleration
		myAcc.x += dist.x * ratio;
		myAcc.y += dist.y * ratio;
		myAcc.z += dist.z * ratio;

		// return updated my-body acceleration
		return myAcc;
	}

	__global__ void updateAcc(const float4* pos_dev, float4* acc_dev)
	{
		// cache shared between threads in a block
		extern __shared__ float4 sharedPos[];

		// NOTE: each thread is assigned a body (denoted "my-body")
		// retrieve my-body data
		unsigned int myIdx = blockIdx.x * BLOCK_SIZE + threadIdx.x;
		float4 myPos = pos_dev[myIdx];
		float4 myAcc = { 0.0f, 0.0f, 0.0f };

		// calculate acceleration of my-body due to all other bodies
		for (unsigned int tileIdx = 0; tileIdx < gridDim.x; tileIdx++) {
			// populate cache (each thread contributes one entry)
			unsigned int otherIdx = tileIdx * BLOCK_SIZE + threadIdx.x;
			sharedPos[threadIdx.x] = pos_dev[otherIdx];

			// synchronize to keep cache data relevant
			__syncthreads();
			for (otherIdx = 0; otherIdx < BLOCK_SIZE; otherIdx++) {
				float4 otherPos = sharedPos[otherIdx];
				myAcc = accFromOther(myPos, otherPos, myAcc);
			}
			__syncthreads();
		}

		// store updated my-body acceleration
		acc_dev[myIdx] = myAcc;
	}

	__global__ void	updatePosAndVel(float4* pos_dev, float4* vel_dev, 
		const float4* acc_dev, float dt)
	{
		// NOTE: each block is assigned a body (denoted "my-body")
		// retrieve my-body data
		unsigned int myIdx = blockIdx.x;
		float4 myPos = pos_dev[myIdx];
		float4 myVel = vel_dev[myIdx];
		float4 myAcc = acc_dev[myIdx];

		// update my-body velocity
		myVel.x += myAcc.x * dt;
		myVel.y += myAcc.y * dt;
		myVel.z += myAcc.z * dt;

		// update my-body position
		myPos.x += myVel.x * dt;
		myPos.y += myVel.y * dt;
		myPos.z += myVel.z * dt;

		// store updated my-body position and velocity
		pos_dev[myIdx] = myPos;
		vel_dev[myIdx] = myVel;
	}
}
