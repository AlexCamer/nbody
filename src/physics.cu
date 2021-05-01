#include <device_launch_parameters.h>
#include "constants.cuh"
#include "physics.cuh"

__device__ float3
body_acc(float3 my_body_pos, float3 my_body_acc, float4 other_body_pos_and_mass)
{
	//
	float3 dist;
	dist.x = other_body_pos_and_mass.x - my_body_pos.x;
	dist.y = other_body_pos_and_mass.y - my_body_pos.y;
	dist.z = other_body_pos_and_mass.z - my_body_pos.z;

	//
	float temp;
	temp = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z + NAN_GUARD;
	temp = temp * temp * temp;
	float dist_ratio = other_body_pos_and_mass.w / sqrtf(temp);

	//
	my_body_acc.x += dist.x * dist_ratio;
	my_body_acc.y += dist.y * dist_ratio;
	my_body_acc.z += dist.z * dist_ratio;
	return my_body_acc;
}

__device__ float3
tile_acc(float3 my_body_pos, float3 my_body_acc)
{
	//
	extern __shared__ float4 shared_pos_and_mass[];

	//
#pragma unroll
	for (unsigned int i = 0; i < TILE_SIZE; i++) {
		my_body_acc = body_acc(my_body_pos, my_body_acc, shared_pos_and_mass[i]);
	}
	return my_body_acc;
}

__global__ void
update_acc(const float3 *pos_dev, float3 *acc_dev, const float *mass_dev, unsigned int n)
{
	//
	extern __shared__ float4 shared_pos_and_mass[];

	//
	int my_body_index = blockIdx.x * TILE_SIZE + threadIdx.x;
	if (my_body_index >= n) {
		return;
	}

	//
	float3 my_body_pos = pos_dev[my_body_index];
	float3 my_body_acc = { 0.0f, 0.0f, 0.0f };

	//
	for (unsigned int i = 0, tile = 0; i < n; i += TILE_SIZE, tile++) {
		unsigned int other_body_index = tile * TILE_SIZE + threadIdx.x;
		float3 other_body_pos = pos_dev[other_body_index];
		shared_pos_and_mass[threadIdx.x] = {
			other_body_pos.x,          // x
			other_body_pos.y,          // y
			other_body_pos.z,          // z
			mass_dev[other_body_index] // w
		};
		__syncthreads();
		my_body_acc = tile_acc(my_body_pos, my_body_acc);
		__syncthreads();
	}

	//
	acc_dev[my_body_index] = my_body_acc;
}

__global__ void
update_pos_and_vel(float3 *pos_dev, float3 *vel_dev, const float3 *acc_dev)
{
	//
	unsigned int my_body_index = blockIdx.x;

	//
	float3 my_body_pos = pos_dev[my_body_index];
	float3 my_body_vel = vel_dev[my_body_index];
	float3 my_body_acc = acc_dev[my_body_index];

	//
	my_body_pos.x += my_body_vel.x * DT + my_body_acc.x * DT * DT / 2;
	my_body_pos.y += my_body_vel.y * DT + my_body_acc.y * DT * DT / 2;
	my_body_pos.z += my_body_vel.z * DT + my_body_acc.z * DT * DT / 2;
	pos_dev[my_body_index] = my_body_pos;

	//
	my_body_vel.x += my_body_acc.x * DT;
	my_body_vel.y += my_body_acc.y * DT;
	my_body_vel.z += my_body_acc.z * DT;
	vel_dev[my_body_index] = my_body_vel;
}
