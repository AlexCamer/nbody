#include <iostream>
#include <cstdlib>
#include <ctime>

#include <cuda_runtime.h>

#include "universe.h"

// benchmark parameters
constexpr size_t NUM_BODIES = 8192;
constexpr unsigned int NUM_STEPS = 1000;
constexpr float DT = 0.01f;

/**
 * Retrieves a random float in range [0, 1].
 * @return  Random float.
 */
float randomFloat(void) {
	return (float)rand() / (float)RAND_MAX;
}

/**
 * Populates a universe with random bodies.
 * @param univ  Universe to populate.
 */
void populateUniverse(nbody::Universe& univ)
{
	srand((unsigned)time(0));
	for (int i = 0; i < NUM_BODIES; i++) {
		float pos[3] = { randomFloat(), randomFloat(), randomFloat() };
		float vel[3] = { randomFloat(), randomFloat(), randomFloat() };
		float mass = randomFloat();
		univ.add(pos, vel, mass);
	}
}

/**
 * Repeatedly steps through a generated universe.
 * @return  Error status.
 */
int main(void)
{
	// setup
	nbody::Universe univ{};
	populateUniverse(univ);

	std::cout << "Started computation.\n";
	clock_t start = clock();

	// continuously step through universe
	for (unsigned int i = 1; i <= NUM_STEPS; i++) {
		univ.step(DT);
	}

	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	float fps = NUM_STEPS / seconds;
	std::cout << "Finished computation with " << fps << " fps.\n";

	return 0;
}
