#include <stdio.h>
#include "constants.cuh"
#include "physics.cuh"
#include "universe.cuh"

__host__ struct universe *
universe_create(const float3 *pos_host, const float3 *vel_host, const float *mass_host, unsigned int n)
{
	cudaError_t err;

	// allocate memory on host for universe struct
	struct universe *univ;
	err = cudaMallocHost((void **)&univ, sizeof(struct universe));
	if (err) {
		fprintf(stderr, "Failed to allocate memory on host for universe struct.");
		return NULL;
	}

	univ->pos_dev = NULL;
	univ->vel_dev = NULL;
	univ->acc_dev = NULL;
	univ->mass_dev = NULL;
	univ->n = n;

	// allocate memory on device for position vector
	err = cudaMalloc((void **)&univ->pos_dev, n * sizeof(float3));
	if (err) {
		fprintf(stderr, "Failed to allocate memory on device for position vector.");
		goto cleanup;
	}

	// allocate memory on device for velocity vector
	err = cudaMalloc((void **)&univ->vel_dev, n * sizeof(float3));
	if (err) {
		fprintf(stderr, "Failed to allocate memory on device for velocity vector.");
		goto cleanup;
	}

	// allocate memory on device for acceleration vector
	err = cudaMalloc((void **)&univ->acc_dev, n * sizeof(float3));
	if (err) {
		fprintf(stderr, "Failed to allocate memory on device for acceleration vector.");
		goto cleanup;
	}

	// allocate memory on device for mass vector
	err = cudaMalloc((void **)&univ->mass_dev, n * sizeof(float));
	if (err) {
		fprintf(stderr, "Failed to allocate memory on device for mass vector.");
		goto cleanup;
	}

	// copy position vector from host to device
	err = cudaMemcpy(univ->pos_dev, pos_host, n * sizeof(float3), cudaMemcpyHostToDevice);
	if (err) {
		fprintf(stderr, "Failed to copy position vector from host to device.");
		goto cleanup;
	}

	// copy velocity vector from host to device
	err = cudaMemcpy(univ->vel_dev, vel_host, n * sizeof(float3), cudaMemcpyHostToDevice);
	if (err) {
		fprintf(stderr, "Failed to copy velocity vector from host to device.");
		goto cleanup;
	}

	// copy mass vector from host to device
	err = cudaMemcpy(univ->mass_dev, mass_host, n * sizeof(float), cudaMemcpyHostToDevice);
	if (err) {
		fprintf(stderr, "Failed to copy mass vector from host to device.");
		goto cleanup;
	}

	return univ;

cleanup:
	// destroy universe struct and vectors
	universe_destroy(univ);
	return NULL;
}

__host__ void
universe_destroy(struct universe *univ)
{
	cudaFree(univ->pos_dev);
	cudaFree(univ->vel_dev);
	cudaFree(univ->acc_dev);
	cudaFree(univ->mass_dev);
	cudaFreeHost(univ);
}

__host__ cudaError_t
universe_step(struct universe *univ)
{
	cudaError_t err;

	// update acceleration vector based on position and mass vectors
	update_acc<<<univ->n, 1>>>(univ->pos_dev, univ->acc_dev, univ->mass_dev, univ->n);
	err = cudaGetLastError();
	if (err) {
		fprintf(stderr, "Failed to update acceleration vector.");
		return err;
	}

	// wait for device to finish computation
	err = cudaDeviceSynchronize();
	if (err) {
		fprintf(stderr, "Failed to synchronize host and device.");
		return err;
	}

	// update position and velocity vectors based on the new acceleration vector
	update_pos_and_vel<<<univ->n, 1>>>(univ->pos_dev, univ->vel_dev, univ->acc_dev);
	err = cudaGetLastError();
	if (err) {
		fprintf(stderr, "Failed to update position and velocity vectors.");
		return err;
	}

	// wait for device to finish computation
	err = cudaDeviceSynchronize();
	if (err) {
		fprintf(stderr, "Failed to synchronize host and device.");
		return err;
	}

	return cudaSuccess;
}

__host__ cudaError_t
universe_state(const struct universe *univ, float3 *pos_host)
{
	// copy position vector from host to device
	cudaError_t err = cudaMemcpy(pos_host, univ->pos_dev, univ->n * sizeof(float3), cudaMemcpyDeviceToHost);
	if (err) {
		fprintf(stderr, "Failed to copy position vector from device to host.");
		return err;
	}

	return cudaSuccess;
}
