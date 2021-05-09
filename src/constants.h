#pragma once

#include <cuda_runtime.h>

namespace nbody
{
	// number of threads per block
	constexpr unsigned int BLOCK_SIZE = 256;

	// gravitational constant
	constexpr float G = 6.674e-11f;

	// small number to prevent division by zero
	constexpr float NAN_GUARD = 1e-10f;
}
