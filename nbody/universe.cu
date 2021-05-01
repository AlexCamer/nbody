#include <stdio.h>
#include "constants.cuh"
#include "physics.cuh"
#include "universe.cuh"

struct universe* universe_create(float3* pos_host, float3* vel_host, float* mass_host, unsigned int n) {
	cudaError_t cuda_status;

	// allocate host memory for universe struct
	struct universe* univ;
	cuda_status = cudaMallocHost((void**)univ, sizeof(struct universe));
	if (univ == NULL) {
		fprintf(stderr, "Failed to allocate host memory for universe struct.");
		return NULL;
	}

	univ->pos_dev = NULL;
	univ->vel_dev = NULL;
	univ->acc_dev = NULL;
	univ->mass_dev = NULL;
	univ->n = n;

	// allocate device memory for position vector
	cuda_status = cudaMalloc((void**)univ->pos_dev, n * sizeof(float3));
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory for position vector.");
		goto error;
	}

	// allocate device memory for velocity vector
	cuda_status = cudaMalloc((void**)univ->vel_dev, n * sizeof(float3));
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory for velocity vector.");
		goto error;
	}

	// allocate device memory for acceleration vector
	cuda_status = cudaMalloc((void**)univ->acc_dev, n * sizeof(float3));
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory for acceleration vector.");
		goto error;
	}

	// allocate device memory for mass vector
	cuda_status = cudaMalloc((void**)univ->mass_dev, n * sizeof(float));
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "Failed to allocate device memory for mass vector.");
		goto error;
	}

	// copy position vector from host to device
	cuda_status = cudaMemcpy(pos_host, univ->pos_dev, n * sizeof(float3), cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "Failed to copy position vector from host to device.");
		goto error;
	}

	// copy velocity vector from host to device
	cuda_status = cudaMemcpy(vel_host, univ->vel_dev, n * sizeof(float3), cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "Failed to copy velocity vector from host to device.");
		goto error;
	}

	// copy mass vector from host to device
	cuda_status = cudaMemcpy(mass_host, univ->mass_dev, n * sizeof(float), cudaMemcpyHostToDevice);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "Failed to copy mass vector from host to device.");
		goto error;
	}

	return univ;

error:
	// destroy universe struct and vectors
	universe_destroy(univ);
	return NULL;
}

void universe_destroy(struct universe* univ) {
	cudaFree(univ->pos_dev);
	cudaFree(univ->vel_dev);
	cudaFree(univ->acc_dev);
	cudaFree(univ->mass_dev);
	cudaFreeHost(univ);
}

int universe_step(struct universe* univ) {
	cudaError_t cuda_status;

	// update acceleration vector based on position and mass vectors
	unsigned int num_blocks = (univ->n + TILE_SIZE - 1) / TILE_SIZE; // equivalent to ROUND_UP(univ->n / TILE_SIZE)
	unsigned int shared_mem_size = univ->n * sizeof(float4);
	update_acc<<<num_blocks, TILE_SIZE, shared_mem_size>>>(univ->pos_dev, univ->acc_dev, univ->mass_dev, univ->n);
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "Failed to update acceleration vector on device.");
		return 1;
	}

	// wait for device to finish computation
	cuda_status = cudaDeviceSynchronize();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "Failed to synchronize host and device.");
		return 1;
	}

	// update position and velocity vectors based on the new acceleration vector
	update_pos_and_vel<<<univ->n, 1>>>(univ->pos_dev, univ->vel_dev, univ->acc_dev);
	cuda_status = cudaGetLastError();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "Failed to update position and velocity vectors on device.");
		return 1;
	}

	// wait for device to finish computation
	cuda_status = cudaDeviceSynchronize();
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "Failed to synchronize host and device.");
		return 1;
	}

	return 0;
}

int universe_state(struct universe* univ, float3* pos_host) {
	// copy position vector from host to device
	cudaError_t cuda_status = cudaMemcpy(pos_host, univ->pos_dev, univ->n * sizeof(float3), cudaMemcpyDeviceToHost);
	if (cuda_status != cudaSuccess) {
		fprintf(stderr, "Failed to copy position vector from device to host.");
		return 1;
	}
	return 0;
}
