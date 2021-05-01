#include <stdio.h>
#include "universe.cuh"

#define NUM_BODIES 100
#define LIFE_SPAN 10000

int
main()
{
    struct universe* univ = universe_create(NULL, NULL, NULL, 0);
    if (univ == NULL) {
        fprintf(stderr, "Universe creation failed.");
        return 1;
    }

    for (unsigned int i = 0; i < LIFE_SPAN; i++) {
        universe_step(univ);
    }

    universe_destroy(univ);
    return 0;

error:
    universe_destroy(univ);
    return 1;
}
