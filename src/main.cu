#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "universe.cuh"

#define NUM_BODIES 250
#define LIFE_SPAN 10000
#define VERBOSE false

__host__ float
random_float() {
	return (float)rand() / (float)RAND_MAX;
}

__host__ void
populate_state(float3 *pos_host, float3 *vel_host, float *mass_host)
{
	srand((unsigned)time(NULL));
	for (int i = 0; i < NUM_BODIES; i++) {
		pos_host[i] = { random_float(), random_float(), random_float() };
		vel_host[i] = { random_float(), random_float(), random_float() };
		mass_host[i] = random_float();
	}
}

__host__ void
print_state(float3 *pos_host, int state_num)
{
	printf("State %d:\n", state_num);
	float3 cur_pos;
	for (int i = 0; i < NUM_BODIES; i++) {
		cur_pos = pos_host[i];
		printf("\tBody %d: (%.2f, %.2f, %.2f)\n", i, cur_pos.x, cur_pos.y, cur_pos.z);
	}
	printf("\n");
}

__host__ int
main()
{
	cudaError_t err;

	printf("Started memory allocation.\n");
		float3* pos_host = NULL;
		float3 *vel_host = NULL;
		float *mass_host = NULL;

		// allocate memory (on host) for position vector
		err = cudaMallocHost((void **)&pos_host, NUM_BODIES * sizeof(float3));
		if (err) {
			fprintf(stderr, "Failed to allocate memory (on host) for position vector.");
			goto cleanup;
		}

		// allocate memory (on host) for velocity vector
		err = cudaMallocHost((void **)&vel_host, NUM_BODIES * sizeof(float3));
		if (err) {
			fprintf(stderr, "Failed to allocate memory (on host) for velocity vector.");
			goto cleanup;
		}

		// allocate memory (on host) for mass vector
		err = cudaMallocHost((void **)&mass_host, NUM_BODIES * sizeof(float));
		if (err) {
			fprintf(stderr, "Failed to allocate memory (on host) for mass vector.");
			goto cleanup;
		}

		populate_state(pos_host, vel_host, mass_host);

		// create universe struct
		struct universe* univ = universe_create(pos_host, vel_host, mass_host, NUM_BODIES);
		if (univ == NULL) {
			fprintf(stderr, "Failed to create universe struct.");
			goto cleanup;
		}
	printf("Finished memory allocation.\n");

	printf("Started computation.\n");
		clock_t start = clock();

		// retrieve initial universe state
		err = universe_state(univ, pos_host);
		if (err) {
			fprintf(stderr, "Failed to retrieve universe state.");
			goto cleanup;
		}

		// print initial state
		if (VERBOSE) {
			print_state(pos_host, 0);
		}

		for (unsigned int i = 1; i <= LIFE_SPAN; i++) {
			// update universe state
			err = universe_step(univ);
			if (err) {
				fprintf(stderr, "Failed to update universe state.");
				goto cleanup;
			}
			
			// retrieve current universe state
			err = universe_state(univ, pos_host);
			if (err) {
				fprintf(stderr, "Failed to retrieve universe state.");
				goto cleanup;
			}

			// print current state
			if (VERBOSE) {
				print_state(pos_host, i);
			}
		}

		clock_t end = clock();
		float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	printf("Finished computation in %.2f seconds.\n", seconds);

cleanup:
	printf("Started cleanup.\n");
		cudaFreeHost(pos_host);
		cudaFreeHost(vel_host);
		cudaFreeHost(mass_host);
		if (univ != NULL) {
			universe_destroy(univ);
		}
	printf("Finished cleanup.\n");
	return 0;
}
