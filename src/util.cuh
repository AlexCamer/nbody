#pragma once

#include <cuda_runtime.h>
#include <iostream>

#define cudaCheckError(err) __cudaCheckError(err, __FILE__, __LINE__)

inline void __cudaCheckError(cudaError_t err, const char* file,
	const int line)
{
	if (err) {
		std::cerr << "Cuda error occured in \"" << file;
		std::cerr <<  "\" at line " << line << ".\n";
		exit(-1);
	}
}

