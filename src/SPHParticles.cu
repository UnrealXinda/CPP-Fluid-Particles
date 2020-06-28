#include <vector>
#include <memory>
#include <cuda_runtime.h>
#include <helper_math.h>
#include <thrust/sort.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/execution_policy.h>
#include "CUDAFunctions.cuh"
#include "DArray.h"
#include "Particles.h"
#include "SPHParticles.h"

__global__ void populateTransforms_CUDA(float* mats, float3* pos, const int num)
{
	const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (i < num)
	{
		float* mat = &mats[i * 16];
		float3 newPos = pos[i] * 1000;
		int idx = 0;

		//Mat.M[0][0] = 1.0f; Mat.M[0][1] = 0.0f; Mat.M[0][2] = 0.0f; Mat.M[0][3] = 0.0f;
		//Mat.M[1][0] = 0.0f; Mat.M[1][1] = 1.0f; Mat.M[1][2] = 0.0f; Mat.M[1][3] = 0.0f;
		//Mat.M[2][0] = 0.0f; Mat.M[2][1] = 0.0f; Mat.M[2][2] = 1.0f; Mat.M[2][3] = 0.0f;
		//Mat.M[3][0] = X;    Mat.M[3][1] = Y;    Mat.M[3][2] = Z;    Mat.M[3][3] = 1.0f;

		mat[idx++] = 1.0f;     mat[idx++] = 0.0f;     mat[idx++] = 0.0f;     mat[idx++] = 0.0f;
		mat[idx++] = 0.0f;     mat[idx++] = 1.0f;     mat[idx++] = 0.0f;     mat[idx++] = 0.0f;
		mat[idx++] = 0.0f;     mat[idx++] = 0.0f;     mat[idx++] = 1.0f;     mat[idx++] = 0.0f;
		mat[idx++] = newPos.z; mat[idx++] = newPos.x; mat[idx++] = newPos.y; mat[idx++] = 1.0f;	// Z up	
	}

	return;
}

__global__ void adjustPositionsToZUp_CUDA(float3* pos, const int num)
{
	const unsigned int i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;

	if (i < num)
	{
		float3 newPos = make_float3(pos[i].y, pos[i].z, pos[i].x);
		pos[i] = newPos;
	}

	return;
}

void SPHParticles::populateTransforms()
{
	populateTransforms_CUDA << <(size() - 1) / block_size + 1, block_size >> > (
		transforms.addr(), pos.addr(), size());
	cudaDeviceSynchronize();
}

void SPHParticles::adjustPositionsToZUp()
{
	thrust::transform(thrust::device,
		pos.addr(), pos.addr() + size(),
		pos.addr(),
		[]__host__ __device__(const float3& p) { return make_float3(p.y, p.z, p.x); }
	);

	//adjustPositionsToZUp_CUDA << <(size() - 1) / block_size + 1, block_size >> > (pos.addr(), size());
	//cudaDeviceSynchronize();
}