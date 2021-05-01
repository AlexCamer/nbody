#ifndef _UNIVERSE_CUH_
#define _UNIVERSE_CUH_

#include <cuda_runtime.h>

struct universe
{
	float3* pos_dev; // position vector (on device)
	float3* vel_dev; // velocity vector (on device)
	float3* acc_dev; // acceleration vector (on device)
	float* mass_dev; // mass vector (on device)
	unsigned int n;  // vector size
};

/**
 * Creates a universe struct.
 *
 * @param pos_host   Position vector (on host) to copy from.
 * @param vel_host   Velocity vector (on host) to copy from.
 * @param mass_host  Mass vector (on host) to copy from.
 * @param n          Vector size.
 * @return           Universe struct.
 */
struct universe*
universe_create(float3* pos_host, float3* vel_host, float* mass_host, unsigned int n);

/**
 * Destroys a universe struct.
 *
 * @param univ  Universe struct to destroy.
 */
void
universe_destroy(struct universe* univ);

/**
 * Performs one universal time step.
 *
 * @param univ  Universe struct to update.
 * @return      Error status.
 */
int
universe_step(struct universe* univ);

/**
 * Retrieves universe state in terms of body positions.
 *
 * @param univ      Universe struct to retrieve state of.
 * @param pos_host  Position vector (on host) to copy to.
 * @return          Error status.
 */
int
universe_state(struct universe* univ, float3* pos_host);

#endif
