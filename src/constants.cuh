#pragma once

namespace nbody
{
	// gravitational constant
	constexpr float G = 6.674e-11f;

	// small number to prevent division by zero
	constexpr float NAN_GUARD = 1e-10f;
}
