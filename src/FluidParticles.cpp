#include <memory>

#include "FluidParticles.h"
#include "SPHParticles.h"

struct FSPHParticles::FImpl
{
	FImpl(float* pos, int size) :
		pParticles(std::make_shared<SPHParticles>(pos, size))
	{}

	std::shared_ptr<SPHParticles> pParticles;
};

FSPHParticles::FSPHParticles(float* pos, int size)
{
	Impl = new FSPHParticles::FImpl(pos, size);
}

FSPHParticles::~FSPHParticles()
{
	delete Impl;
}

std::shared_ptr<class SPHParticles> FSPHParticles::GetParticlesPtr() const
{
	return Impl->pParticles;
}

int FSPHParticles::Size() const
{
	return Impl->pParticles->size();
}

void FSPHParticles::CopyBackParticlePositions(float* Dst)
{
	CUDA_CALL(cudaMemcpy(Dst, Impl->pParticles->getPosPtr(), sizeof(float) * 3 * Size(), cudaMemcpyDeviceToHost));
}

void FSPHParticles::CopyBackParticleTransforms(float* Dst)
{
	Impl->pParticles->populateTransforms();
	CUDA_CALL(cudaMemcpy(Dst, Impl->pParticles->getTransformsPtr(), sizeof(float) * 16 * Size(), cudaMemcpyDeviceToHost));
}

struct FBoundaryParticles::FImpl
{
	FImpl(float* pos, int size) :
		pParticles(std::make_shared<SPHParticles>(pos, size))
	{}

	std::shared_ptr<SPHParticles> pParticles;
};

FBoundaryParticles::FBoundaryParticles(float* pos, int size)
{
	Impl = new FBoundaryParticles::FImpl(pos, size);
}

FBoundaryParticles::~FBoundaryParticles()
{
	delete Impl;
}

std::shared_ptr<class SPHParticles> FBoundaryParticles::GetParticlesPtr() const
{
	return Impl->pParticles;
}

int FBoundaryParticles::Size() const
{
	return Impl->pParticles->size();
}