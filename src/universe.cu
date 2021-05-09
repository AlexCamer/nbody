#include <cuda_runtime.h>

#include "constants.h"
#include "physics.cuh"
#include "universe.h"
#include "util.cuh"

namespace nbody
{
	struct Universe::Data
	{
		float4* pos_dev;               // position-and-mass vector (on device)
		float4* vel_dev;               // velocity vector (on device)
		float4* acc_dev;               // acceration vector (on device)
		size_t size = 0;               // vectors size
		size_t capacity = BLOCK_SIZE;  // vectors capacity
	};

	Universe::Universe(void) : data{ new Data{} }
	{
		size_t initialMemSize = data->capacity * sizeof(float4);

		// allocate memory for vectors
		cudaCheckError(cudaMalloc(&data->pos_dev, initialMemSize));
		cudaCheckError(cudaMalloc(&data->vel_dev, initialMemSize));
		cudaCheckError(cudaMalloc(&data->acc_dev, initialMemSize));

		// pad position-and-mass vector
		cudaCheckError(cudaMemset(data->pos_dev, 0, initialMemSize));
	}

	Universe::~Universe(void)
	{
		delete data;
	}

	void Universe::add(const float* pos, const float* vel, float mass)
	{
		// double capacity if at limit
		if (data->size == data->capacity) {
			data->capacity *= 2;
			size_t oldMemSize = data->size * sizeof(float4);
			size_t newMemSize = data->capacity * sizeof(float4);

			// grow position-and-mass vector to new capacity
			float4* oldPos_dev = data->pos_dev;
			float4* newPos_dev;
			cudaCheckError(cudaMalloc(&newPos_dev, newMemSize));   // allocate new memory
			cudaCheckError(cudaMemcpy(newPos_dev, oldPos_dev,      // copy from old memory
				oldMemSize, cudaMemcpyDeviceToDevice));
			cudaCheckError(cudaMemset(newPos_dev + data->size, 0,  // pad new memory
				newMemSize - oldMemSize));
			cudaCheckError(cudaFree(oldPos_dev));                  // free old memory
			data->pos_dev = newPos_dev;

			// grow velocity vector to new capacity
			float4* oldVel_dev = data->vel_dev;
			float4* newVel_dev;
			cudaCheckError(cudaMalloc(&newVel_dev, newMemSize));  // allocate new memory
			cudaCheckError(cudaMemcpy(newVel_dev, oldVel_dev,     // copy from old memory
				oldMemSize, cudaMemcpyDeviceToDevice));
			cudaCheckError(cudaFree(oldVel_dev));                 // free old memory
			data->vel_dev = newVel_dev;

			// grow acceration vector to new capacity
			float4* oldAcc_dev = data->acc_dev;
			float4* newAcc_dev;
			cudaCheckError(cudaMalloc(&newAcc_dev, newMemSize));  // allocate new memory
			cudaCheckError(cudaFree(oldAcc_dev));                 // free old memory
			data->acc_dev = newAcc_dev;
		}

		// add position, mass, and velocity to vectors
		// NOTE: store mass with position for faster read
		float4 posToAdd = { pos[0], pos[1], pos[2], mass };
		float4 velToAdd = { vel[0], vel[1], vel[2] };
		cudaCheckError(cudaMemcpy(data->pos_dev + data->size,     // copy to memory
			&posToAdd, sizeof(float4), cudaMemcpyHostToDevice));
		cudaCheckError(cudaMemcpy(data->vel_dev + data->size,     // copy to memory
			&velToAdd, sizeof(float4), cudaMemcpyHostToDevice));

		// increment size
		data->size++;
	}

	void Universe::reset(void)
	{
		size_t memSize = data->capacity * sizeof(float4);
		cudaCheckError(cudaMemset(data->pos_dev, 0, memSize));  // pad memory
		data->size = 0;
	}

	size_t Universe::size(void)
	{
		return data->size;
	}

	void Universe::step(float dt)
	{
		unsigned int numBlocks;
		unsigned int sharedMemSize;

		// update accerations based on positions and masses
		numBlocks = (data->size + BLOCK_SIZE - 1) / BLOCK_SIZE;
		sharedMemSize = BLOCK_SIZE * sizeof(float4);
		updateAcc<<<numBlocks, BLOCK_SIZE, sharedMemSize>>>(
			data->pos_dev, data->acc_dev);

		// wait for device to finish computation
		cudaDeviceSynchronize();

		// update positions and velocities based on new accerations
		numBlocks = data->size;
		updatePosAndVel<<<numBlocks, 1>>>(data->pos_dev,
			data->vel_dev, data->acc_dev, dt);

		// wait for device to finish computation
		cudaDeviceSynchronize();
	}
}
