#include <device_launch_parameters.h>
#include "constants.cuh"
#include "physics.cuh"

__global__ void
update_acc(const float3* pos_dev, float3* acc_dev, const float* mass_dev, unsigned int n)
{
	// get index of body assigned to block (my_body)
	int my_body_index = blockIdx.x;

	// recover my_body position
	float3 my_body_pos = pos_dev[my_body_index];;
	float3 my_body_acc = { 0.0f, 0.0f, 0.0f };

	// calculate acceleration of my_body due to all other bodies
	for (unsigned int i = 0; i < n; i++) {
		float3 other_body_pos = pos_dev[i];

		// calculate distance vector between my_body and other_body
		float3 dist;
		dist.x = other_body_pos.x - my_body_pos.x;
		dist.y = other_body_pos.y - my_body_pos.y;
		dist.z = other_body_pos.z - my_body_pos.z;

		// calculate ratio between acceleration and distance
		float temp = dist.x * dist.x + dist.y * dist.y + dist.z * dist.z + NAN_GUARD;
		temp = temp * temp * temp;
		float ratio = mass_dev[i] / sqrtf(temp);

		// add acceleration due to other_body to my_body acceleration
		my_body_acc.x += dist.x * ratio;
		my_body_acc.y += dist.y * ratio;
		my_body_acc.z += dist.z * ratio;
	}

	// store new acceleration
	acc_dev[my_body_index] = my_body_acc;
}

__global__ void
update_pos_and_vel(float3 *pos_dev, float3 *vel_dev, const float3 *acc_dev)
{
	// get index of body assigned to block (my_body)
	unsigned int my_body_index = blockIdx.x;

	// recover my_body position, velocity, and acceleration
	float3 my_body_pos = pos_dev[my_body_index];
	float3 my_body_vel = vel_dev[my_body_index];
	float3 my_body_acc = acc_dev[my_body_index];

	// update my_body position
	my_body_pos.x += my_body_vel.x * DT + my_body_acc.x * DT * DT / 2;
	my_body_pos.y += my_body_vel.y * DT + my_body_acc.y * DT * DT / 2;
	my_body_pos.z += my_body_vel.z * DT + my_body_acc.z * DT * DT / 2;
	pos_dev[my_body_index] = my_body_pos;

	// update my_body velocity
	my_body_vel.x += my_body_acc.x * DT;
	my_body_vel.y += my_body_acc.y * DT;
	my_body_vel.z += my_body_acc.z * DT;
	vel_dev[my_body_index] = my_body_vel;
}
