#pragma once

namespace nbody
{
	struct __declspec(dllexport) Universe
	{
		/**
		 * Universe constructor.
		 */
		Universe();

		/**
		 * Universe destructor.
		 */
		virtual
		~Universe();

		/**
		 * Adds a body to the universe.
		 * @param pos   Body position array (form: [x, y, z]).
		 * @param vel   Body Velocity array (form: [x, y, z]).
		 * @param mass  Body Mass.
		 */
		void
		add(const float* pos, const float* vel, float mass);

		/**
		 * Removes a body from the universe.
		 * @param bodyIdx  Body index.
		 */
		void
		remove(unsigned int bodyIdx);

		/**
		 * Removes all bodies from the universe.
		 */
		void
		reset();

		/**
		 * Retrieves number of bodies in the universe.
		 * @return  Number of bodies.
		 */
		size_t
		size();

		/**
		 * Performs one universal time step.
		 * @param dt  Time increment.
		 */
		void
		step(float dt);

		/**
		 * Retrieves position of a body.
		 * @param bodyIdx  Body index.
		 * @return         Body position array (form: [x, y, z]).
		 */
		const float*
		position(unsigned int bodyIdx);

		/**
		 * Retrieves velocity of a body.
		 * @param bodyIdx  Body index.
		 * @return         Body velocity array (form: [x, y, z]).
		 */
		const float*
		velocity(unsigned int bodyIdx);

		/**
		 * Retrieves mass of a body.
		 * @param bodyIdx  Body index.
		 * @return         Body mass.
		 */
		float
		mass(unsigned int bodyIdx);

	private:
		struct Data;
		Data* data;
	};
}
