#include <iostream>
#include <cstdlib>
#include <ctime>

#include "universe.h"

// benchmark parameters
constexpr size_t NUM_BODIES = 250;
constexpr unsigned int NUM_STEPS = 10000;
constexpr float DT = 0.01f;
constexpr bool VERBOSE = false;

/**
 * Retrieves a random float in range [0, 1].
 * @return  Random float.
 */
float
randomFloat() {
	return (float)rand() / (float)RAND_MAX;
}

/**
 * Populates a universe with random bodies.
 * @param univ  Universe to populate.
 */
void
populateUniverse(nbody::Universe& univ)
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
 * Prints universe state in terms of body positions.
 * @param univ      Universe to print.
 * @param stateNum  State number.
 */
void
printUniverse(nbody::Universe& univ, unsigned int stateNum)
{
	std::cout << "State " << stateNum << ":\n";
	for (int i = 0; i < univ.size(); i++) {
		const float* pos = univ.position(i);
		std::cout << "\tBody " << i << ": ";
		std::cout << "(" << pos[0] << ", " << pos[1] << ", " << pos[2] << ")\n";
	}
	std::cout << "\n";
}

/**
 * Repeatedly steps through a generated universe.
 * @return  Error status.
 */
int
main()
{
	// setup
	nbody::Universe univ{};
	populateUniverse(univ);

	std::cout << "Started computation.\n";
	clock_t start = clock();

	if (VERBOSE) {
		// print initial state
		printUniverse(univ, 0);
	}

	for (unsigned int i = 1; i <= NUM_STEPS; i++) {
		// update universe state
		univ.step(DT);

		if (VERBOSE) {
			// print current state
			printUniverse(univ, i);
		}
	}

	clock_t end = clock();
	float seconds = (float)(end - start) / CLOCKS_PER_SEC;
	std::cout << "Finished computation in " << seconds << " seconds.\n";

	return 0;
}
