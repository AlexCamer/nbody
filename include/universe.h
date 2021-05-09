#pragma once

namespace nbody
{
	struct __declspec(dllexport) Universe
	{
		/**
		 * Universe constructor.
		 */
		Universe(void);

		/**
		 * Universe destructor.
		 */
		virtual ~Universe(void);

		/**
		 * Adds a body to the universe.
		 * @param pos   Body position triple (on host).
		 * @param vel   Body velocity triple (on host).
		 * @param mass  Body mass.
		 */
		void add(const float* pos, const float* vel, float mass);

		/**
		 * Removes all bodies from the universe.
		 */
		void reset(void);

		/**
		 * Retrieves number of bodies in the universe.
		 * @return  Number of bodies.
		 */
		size_t size(void);

		/**
		 * Performs one universal time step.
		 * @param dt  Time increment.
		 */
		void step(float dt);

	private:
		struct Data;
		Data* data;
	};
}
